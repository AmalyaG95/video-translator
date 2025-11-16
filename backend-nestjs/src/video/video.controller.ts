import {
  Controller,
  Post,
  Get,
  Head,
  Delete,
  Param,
  HttpCode,
  HttpStatus,
  NotFoundException,
  StreamableFile,
  Res,
  Req,
  Body,
  Query,
} from '@nestjs/common';
import type { Request, Response } from 'express';
import { createReadStream, existsSync, statSync } from 'fs';
import * as path from 'path';
import { VideoService } from './video.service';
import { SessionsService } from '../sessions/sessions.service';

@Controller()
export class VideoController {
  constructor(
    private readonly videoService: VideoService,
    private readonly sessionsService: SessionsService,
  ) {}

  @Post('translate/:id')
  @HttpCode(HttpStatus.ACCEPTED)
  async startTranslation(@Param('id') sessionId: string) {
    console.log(`🟣🟣🟣 [NESTJS CONTROLLER] POST /translate/${sessionId} called`);
    console.log(`   Session ID: ${sessionId}`);
    
    try {
      console.log(`🟣 [NESTJS CONTROLLER] Calling videoService.startTranslation...`);
      await this.videoService.startTranslation(sessionId);
      console.log(`🟣 [NESTJS CONTROLLER] videoService.startTranslation completed`);
      
      return {
        message: 'Translation started',
        sessionId,
      };
    } catch (error) {
      console.error(`❌❌❌ [NESTJS CONTROLLER] Error in startTranslation:`, error);
      console.error(`   Error type: ${error?.constructor?.name || typeof error}`);
      console.error(`   Error message: ${error?.message || String(error)}`);
      throw error;
    }
  }

  @Delete('translate/:id')
  async cancelTranslation(@Param('id') sessionId: string) {
    const success = await this.videoService.cancelTranslation(sessionId);
    return {
      success,
      message: success ? 'Translation cancelled' : 'Failed to cancel',
    };
  }

  @Post('translate/:id/control')
  @HttpCode(HttpStatus.OK)
  async controlTranslation(
    @Param('id') sessionId: string,
    @Body() body: { action: 'pause' | 'resume' | 'cancel' },
  ) {
    const result = await this.videoService.controlTranslation(
      sessionId,
      body.action,
    );
    return {
      success: result.success,
      isPaused: result.isPaused,
      message: result.message,
    };
  }

  @Get('result/:id')
  async getResult(@Param('id') sessionId: string) {
    return await this.videoService.getResult(sessionId);
  }

  @Get('stream/:id')
  async streamVideo(
    @Param('id') sessionId: string,
    @Res({ passthrough: true }) res: Response,
  ): Promise<StreamableFile> {
    let filePath: string | undefined;

    // Try to get session first (may throw if not found)
    try {
      const session = this.sessionsService.findOne(sessionId);
      filePath = session?.outputPath;
    } catch (error) {
      // Session not found, will try fallback
    }

    // Fallback to checking artifacts directory directly
    if (!filePath || !existsSync(filePath)) {
      // Check environment variable first (for standalone AppImage)
      const artifactsDir = process.env.ARTIFACTS_DIR;
      const dockerPath = `/app/artifacts/${sessionId}_translated.mp4`;
      const localPath = `../.data/artifacts/${sessionId}_translated.mp4`; // Go up one level from backend-nestjs to project root .data folder
      const envPath = artifactsDir ? path.join(artifactsDir, `${sessionId}_translated.mp4`) : null;

      if (envPath && existsSync(envPath)) {
        filePath = envPath;
      } else if (existsSync(dockerPath)) {
        filePath = dockerPath;
      } else if (existsSync(localPath)) {
        filePath = localPath;
      } else {
        throw new NotFoundException('Video file not found');
      }
    }

    const file = createReadStream(filePath);

    res.set({
      'Content-Type': 'video/mp4',
      'Content-Disposition': `inline; filename="translated_${sessionId}.mp4"`,
      'Accept-Ranges': 'bytes',
      'Cache-Control': 'no-cache',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Range',
      'Access-Control-Expose-Headers': 'Content-Range, Content-Length',
    });

    return new StreamableFile(file);
  }

  @Get('translate/:id/segments')
  async getSegments(
    @Param('id') sessionId: string,
    @Query('segmentId') segmentId?: string,
    @Query('download') download?: string,
    @Res({ passthrough: true }) res?: Response,
  ) {
    if (segmentId && download === 'true') {
      // Download specific segment
      const segmentPath = await this.videoService.getSegmentPath(
        sessionId,
        segmentId,
      );
      if (!segmentPath || !existsSync(segmentPath)) {
        throw new NotFoundException('Segment not found');
      }

      const file = createReadStream(segmentPath);
      res!.set({
        'Content-Type': 'video/mp4',
        'Content-Disposition': `inline; filename="segment_${segmentId}.mp4"`,
      });

      return new StreamableFile(file);
    } else {
      // List available segments
      const segments = await this.videoService.getAvailableSegments(sessionId);
      return { segments };
    }
  }

  @Head('download/:id')
  headVideo(@Param('id') sessionId: string, @Res() res: Response): void {
    this.handleHeadRequest(sessionId, res);
  }

  @Get('download/:id')
  downloadVideo(
    @Param('id') sessionId: string,
    @Req() req: Request,
    @Res() res: Response,
  ): void {
    let filePath: string | undefined;

    // Try to get session first (may throw if not found)
    try {
      const session = this.sessionsService.findOne(sessionId);
      filePath =
        session?.result?.outputPath ||
        session?.outputPath ||
        session?.metadata?.outputPath;
    } catch (error) {
      // Session not found, will try fallback
    }

    // Fallback to checking artifacts directory directly
    if (!filePath || !existsSync(filePath)) {
      // Check environment variable first (for standalone AppImage)
      const artifactsDir = process.env.ARTIFACTS_DIR;
      const dockerPath = `/app/artifacts/${sessionId}_translated.mp4`;
      const localPath = path.join(
        process.cwd(),
        '.data',
        'artifacts',
        `${sessionId}_translated.mp4`,
      );
      const altLocalPath = `../.data/artifacts/${sessionId}_translated.mp4`;
      const envPath = artifactsDir ? path.join(artifactsDir, `${sessionId}_translated.mp4`) : null;

      if (envPath && existsSync(envPath)) {
        filePath = envPath;
      } else if (existsSync(dockerPath)) {
        filePath = dockerPath;
      } else if (existsSync(localPath)) {
        filePath = localPath;
      } else if (existsSync(altLocalPath)) {
        filePath = altLocalPath;
      } else {
        console.error(
          `Video file not found for session ${sessionId}. Checked:`,
          {
            dockerPath,
            localPath,
            altLocalPath,
            sessionOutputPath: filePath,
            cwd: process.cwd(),
          },
        );
        throw new NotFoundException(
          `Video file not found for session ${sessionId}`,
        );
      }
    }

    // Check if file exists and get stats
    if (!existsSync(filePath)) {
      console.error(
        `Video file path resolved but file does not exist: ${filePath}`,
      );
      throw new NotFoundException(
        `Video file not found for session ${sessionId}`,
      );
    }

    // Get file stats
    const stats = statSync(filePath);
    const fileSize = stats.size;

    // Set CORS headers first to ensure they're applied
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Range');
    res.setHeader(
      'Access-Control-Expose-Headers',
      'Content-Disposition, Content-Length, Content-Range, Accept-Ranges',
    );

    // Parse Range header for partial content support (essential for large files)
    const range = req.headers.range;
    if (range) {
      // Parse range header (e.g., "bytes=0-1023" or "bytes=1024-")
      const parts = range.replace(/bytes=/, '').split('-');
      const start = parseInt(parts[0], 10);
      const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;
      const chunkSize = end - start + 1;

      // Validate range
      if (start >= fileSize || end >= fileSize || start > end) {
        res.status(416).setHeader('Content-Range', `bytes */${fileSize}`);
        res.end();
        return;
      }

      // Set headers for partial content (206)
      res.status(206); // Partial Content
      res.setHeader('Content-Range', `bytes ${start}-${end}/${fileSize}`);
      res.setHeader('Accept-Ranges', 'bytes');
      res.setHeader('Content-Length', chunkSize.toString());
      res.setHeader('Content-Type', 'video/mp4');
      res.setHeader(
        'Content-Disposition',
        `attachment; filename="translated_${sessionId}.mp4"`,
      );
      res.setHeader('Cache-Control', 'no-cache');

      // Create stream for the requested range
      const file = createReadStream(filePath, { start, end });

      file.on('error', (error) => {
        console.error('Stream error:', error);
        if (!res.headersSent) {
          res.status(500).json({ error: 'Failed to stream video' });
        } else {
          res.destroy();
        }
      });

      file.pipe(res);

      res.on('close', () => {
        if (!file.destroyed) {
          file.destroy();
        }
      });
    } else {
      // No range requested - send full file but still stream it
      res.setHeader('Content-Type', 'video/mp4');
      res.setHeader('Content-Length', fileSize.toString());
      res.setHeader(
        'Content-Disposition',
        `attachment; filename="translated_${sessionId}.mp4"`,
      );
      res.setHeader('Accept-Ranges', 'bytes');
      res.setHeader('Cache-Control', 'no-cache');

      // Create stream and pipe directly to response
      const file = createReadStream(filePath);

      file.on('error', (error) => {
        console.error('Stream error:', error);
        if (!res.headersSent) {
          res.status(500).json({ error: 'Failed to stream video' });
        } else {
          res.destroy();
        }
      });

      file.pipe(res);

      res.on('close', () => {
        if (!file.destroyed) {
          file.destroy();
        }
      });
    }
  }

  // Helper method to handle HEAD requests
  private handleHeadRequest(sessionId: string, res: Response): void {
    let filePath: string | undefined;

    // Try to get session first
    try {
      const session = this.sessionsService.findOne(sessionId);
      filePath =
        session?.result?.outputPath ||
        session?.outputPath ||
        session?.metadata?.outputPath;
    } catch (error) {
      // Session not found, will try fallback
    }

    // Fallback to checking artifacts directory directly
    if (!filePath || !existsSync(filePath)) {
      // Check environment variable first (for standalone AppImage)
      const artifactsDir = process.env.ARTIFACTS_DIR;
      const dockerPath = `/app/artifacts/${sessionId}_translated.mp4`;
      const localPath = path.join(
        process.cwd(),
        '.data',
        'artifacts',
        `${sessionId}_translated.mp4`,
      );
      const altLocalPath = `../.data/artifacts/${sessionId}_translated.mp4`;
      const envPath = artifactsDir ? path.join(artifactsDir, `${sessionId}_translated.mp4`) : null;

      if (envPath && existsSync(envPath)) {
        filePath = envPath;
      } else if (existsSync(dockerPath)) {
        filePath = dockerPath;
      } else if (existsSync(localPath)) {
        filePath = localPath;
      } else if (existsSync(altLocalPath)) {
        filePath = altLocalPath;
      } else {
        throw new NotFoundException(
          `Video file not found for session ${sessionId}`,
        );
      }
    }

    if (!existsSync(filePath)) {
      throw new NotFoundException(
        `Video file not found for session ${sessionId}`,
      );
    }

    const stats = statSync(filePath);
    const fileSize = stats.size;

    // Set headers for HEAD response
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Range');
    res.setHeader(
      'Access-Control-Expose-Headers',
      'Content-Disposition, Content-Length, Content-Range, Accept-Ranges',
    );
    res.setHeader('Content-Type', 'video/mp4');
    res.setHeader('Content-Length', fileSize.toString());
    res.setHeader(
      'Content-Disposition',
      `attachment; filename="translated_${sessionId}.mp4"`,
    );
    res.setHeader('Accept-Ranges', 'bytes');
    res.setHeader('Cache-Control', 'no-cache');
    res.status(200).end();
  }

  @Head('original/:id')
  headOriginalVideo(@Param('id') sessionId: string, @Res() res: Response): void {
    this.handleHeadRequestForOriginal(sessionId, res);
  }

  @Get('original/:id')
  downloadOriginalVideo(
    @Param('id') sessionId: string,
    @Req() req: Request,
    @Res() res: Response,
  ): void {
    let filePath: string | undefined;

    // Get original file path from session
    let session: any = null;
    try {
      session = this.sessionsService.findOne(sessionId);
      filePath = session?.filePath;
      
      if (filePath) {
        // Normalize the path to handle both absolute and relative paths
        if (!path.isAbsolute(filePath)) {
          filePath = path.resolve(process.cwd(), filePath);
        }
        // Verify the file exists at the resolved path
        if (existsSync(filePath)) {
          // File found at session path - use it
        } else {
          // Path from session doesn't exist - clear it to trigger fallback
          filePath = undefined;
        }
      }
    } catch (error) {
      // Session not found - will try fallback paths below
      console.warn(`Session ${sessionId} not found, attempting fallback file search`);
    }

    // Fallback: if session path doesn't exist or session not found, search uploads directory
    if (!filePath || !existsSync(filePath)) {
      try {
        const uploadsDir = path.join(process.cwd(), '.data', 'uploads');
        if (existsSync(uploadsDir)) {
          const fs = require('fs');
          const files = fs.readdirSync(uploadsDir).filter((f: string) => 
            /\.(mp4|mov|avi|mkv|webm)$/i.test(f)
          );

          if (files.length === 0) {
            console.error(`No video files found in uploads directory: ${uploadsDir}`);
          } else {
            // Strategy 1: If we have session data with fileName, try to match by original filename
            // Note: This might not work if the uploaded file has a generated name
            if (session?.fileName && !filePath) {
              const matchingFile = files.find((f: string) => 
                f.toLowerCase().includes(session.fileName.toLowerCase()) || 
                session.fileName.toLowerCase().includes(f.toLowerCase())
              );
              if (matchingFile) {
                filePath = path.join(uploadsDir, matchingFile);
                console.log(`Matched file by fileName: ${matchingFile}`);
              }
            }

            // Strategy 2: If we have filePath from session, extract filename and match
            if ((!filePath || !existsSync(filePath)) && session?.filePath) {
              const sessionFileName = path.basename(session.filePath);
              const matchingFile = files.find((f: string) => 
                f === sessionFileName || 
                f.toLowerCase().includes(sessionFileName.toLowerCase()) || 
                sessionFileName.toLowerCase().includes(f.toLowerCase())
              );
              if (matchingFile) {
                filePath = path.join(uploadsDir, matchingFile);
                console.log(`Matched file by filePath basename: ${matchingFile}`);
              }
            }

            // Strategy 3: Use the most recently modified file (likely the most recent upload)
            if (!filePath || !existsSync(filePath)) {
              const fileStats = files.map((f: string) => {
                const fullPath = path.join(uploadsDir, f);
                try {
                  return {
                    name: f,
                    path: fullPath,
                    mtime: fs.statSync(fullPath).mtime.getTime(),
                  };
                } catch {
                  return null;
                }
              }).filter((stat: { name: string; path: string; mtime: number } | null): stat is { name: string; path: string; mtime: number } => stat !== null);

              if (fileStats.length > 0) {
                // Sort by modification time (most recent first)
                fileStats.sort((a: { name: string; path: string; mtime: number }, b: { name: string; path: string; mtime: number }) => b.mtime - a.mtime);
                filePath = fileStats[0].path;
                console.log(`Using most recently modified file: ${fileStats[0].name}`);
              }
            }

            // Strategy 4: Last resort - if only one video file exists, use it
            if ((!filePath || !existsSync(filePath)) && files.length === 1) {
              filePath = path.join(uploadsDir, files[0]);
              console.warn(
                `Using fallback: single video file found in uploads for session ${sessionId}: ${files[0]}`,
              );
            }
          }
        }
      } catch (err) {
        console.error(`Error searching uploads directory for session ${sessionId}:`, err);
      }
    }

    if (!filePath || !existsSync(filePath)) {
      const checkedPaths = {
        sessionPath: session?.filePath,
        resolvedPath: filePath,
        uploadsDir: path.join(process.cwd(), '.data', 'uploads'),
        sessionExists: !!session,
      };
      console.error(
        `Original video file not found for session ${sessionId}. Checked:`,
        checkedPaths,
      );
      throw new NotFoundException(
        `Original video file not found for session ${sessionId}`,
      );
    }

    // Get file stats
    const stats = statSync(filePath);
    const fileSize = stats.size;

    // Set CORS headers
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Range');
    res.setHeader(
      'Access-Control-Expose-Headers',
      'Content-Disposition, Content-Length, Content-Range, Accept-Ranges',
    );

    // Parse Range header for partial content support
    const range = req.headers.range;
    if (range) {
      const parts = range.replace(/bytes=/, '').split('-');
      const start = parseInt(parts[0], 10);
      const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;
      const chunkSize = end - start + 1;

      // Validate range
      if (start >= fileSize || end >= fileSize || start > end) {
        res.status(416).setHeader('Content-Range', `bytes */${fileSize}`);
        res.end();
        return;
      }

      // Set headers for partial content (206)
      res.status(206);
      res.setHeader('Content-Range', `bytes ${start}-${end}/${fileSize}`);
      res.setHeader('Accept-Ranges', 'bytes');
      res.setHeader('Content-Length', chunkSize.toString());
      res.setHeader('Content-Type', 'video/mp4');
      res.setHeader(
        'Content-Disposition',
        `inline; filename="original_${sessionId}.mp4"`,
      );
      res.setHeader('Cache-Control', 'no-cache');

      // Create stream for the requested range
      const file = createReadStream(filePath, { start, end });

      file.on('error', (error) => {
        console.error('Stream error:', error);
        if (!res.headersSent) {
          res.status(500).json({ error: 'Failed to stream video' });
        } else {
          res.destroy();
        }
      });

      file.pipe(res);

      res.on('close', () => {
        if (!file.destroyed) {
          file.destroy();
        }
      });
    } else {
      // No range requested - send full file but still stream it
      res.setHeader('Content-Type', 'video/mp4');
      res.setHeader('Content-Length', fileSize.toString());
      res.setHeader(
        'Content-Disposition',
        `inline; filename="original_${sessionId}.mp4"`,
      );
      res.setHeader('Accept-Ranges', 'bytes');
      res.setHeader('Cache-Control', 'no-cache');

      const file = createReadStream(filePath);

      file.on('error', (error) => {
        console.error('Stream error:', error);
        if (!res.headersSent) {
          res.status(500).json({ error: 'Failed to stream video' });
        } else {
          res.destroy();
        }
      });

      file.pipe(res);

      res.on('close', () => {
        if (!file.destroyed) {
          file.destroy();
        }
      });
    }
  }

  // Helper method to handle HEAD requests for original video
  private handleHeadRequestForOriginal(
    sessionId: string,
    res: Response,
  ): void {
    let filePath: string | undefined;

    // Get original file path from session
    let session: any = null;
    try {
      session = this.sessionsService.findOne(sessionId);
      filePath = session?.filePath;
      
      if (filePath) {
        // Normalize the path to handle both absolute and relative paths
        if (!path.isAbsolute(filePath)) {
          filePath = path.resolve(process.cwd(), filePath);
        }
        // Verify the file exists at the resolved path
        if (existsSync(filePath)) {
          // File found at session path - use it
        } else {
          // Path from session doesn't exist - clear it to trigger fallback
          filePath = undefined;
        }
      }
    } catch (error) {
      // Session not found - will try fallback paths below
    }

    // Fallback: if session path doesn't exist or session not found, search uploads directory
    if (!filePath || !existsSync(filePath)) {
      try {
        const uploadsDir = path.join(process.cwd(), '.data', 'uploads');
        if (existsSync(uploadsDir)) {
          const fs = require('fs');
          const files = fs.readdirSync(uploadsDir).filter((f: string) => 
            /\.(mp4|mov|avi|mkv|webm)$/i.test(f)
          );

          if (files.length === 0) {
            console.error(`No video files found in uploads directory: ${uploadsDir}`);
          } else {
            // Strategy 1: If we have session data with fileName, try to match by original filename
            if (session?.fileName && !filePath) {
              const matchingFile = files.find((f: string) => 
                f.toLowerCase().includes(session.fileName.toLowerCase()) || 
                session.fileName.toLowerCase().includes(f.toLowerCase())
              );
              if (matchingFile) {
                filePath = path.join(uploadsDir, matchingFile);
              }
            }

            // Strategy 2: If we have filePath from session, extract filename and match
            if ((!filePath || !existsSync(filePath)) && session?.filePath) {
              const sessionFileName = path.basename(session.filePath);
              const matchingFile = files.find((f: string) => 
                f === sessionFileName || 
                f.toLowerCase().includes(sessionFileName.toLowerCase()) || 
                sessionFileName.toLowerCase().includes(f.toLowerCase())
              );
              if (matchingFile) {
                filePath = path.join(uploadsDir, matchingFile);
              }
            }

            // Strategy 3: Use the most recently modified file (likely the most recent upload)
            if (!filePath || !existsSync(filePath)) {
              const fileStats = files.map((f: string) => {
                const fullPath = path.join(uploadsDir, f);
                try {
                  return {
                    name: f,
                    path: fullPath,
                    mtime: fs.statSync(fullPath).mtime.getTime(),
                  };
                } catch {
                  return null;
                }
              }).filter((stat: { name: string; path: string; mtime: number } | null): stat is { name: string; path: string; mtime: number } => stat !== null);

              if (fileStats.length > 0) {
                // Sort by modification time (most recent first)
                fileStats.sort((a: { name: string; path: string; mtime: number }, b: { name: string; path: string; mtime: number }) => b.mtime - a.mtime);
                filePath = fileStats[0].path;
              }
            }

            // Strategy 4: Last resort - if only one video file exists, use it
            if ((!filePath || !existsSync(filePath)) && files.length === 1) {
              filePath = path.join(uploadsDir, files[0]);
            }
          }
        }
      } catch (err) {
        console.error(`Error searching uploads directory for session ${sessionId}:`, err);
      }
    }

    if (!filePath || !existsSync(filePath)) {
      throw new NotFoundException(
        `Original video file not found for session ${sessionId}`,
      );
    }

    const stats = statSync(filePath);
    const fileSize = stats.size;

    // Set headers for HEAD response
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Range');
    res.setHeader(
      'Access-Control-Expose-Headers',
      'Content-Disposition, Content-Length, Content-Range, Accept-Ranges',
    );
    res.setHeader('Content-Type', 'video/mp4');
    res.setHeader('Content-Length', fileSize.toString());
    res.setHeader(
      'Content-Disposition',
      `inline; filename="original_${sessionId}.mp4"`,
    );
    res.setHeader('Accept-Ranges', 'bytes');
    res.setHeader('Cache-Control', 'no-cache');
    res.status(200).end();
  }

  @Get('early-preview/:id')
  async streamEarlyPreview(
    @Param('id') sessionId: string,
    @Res({ passthrough: true }) res: Response,
  ): Promise<StreamableFile> {
    let filePath: string | undefined;

    // Try to get session first (may throw if not found)
    try {
      const session = this.sessionsService.findOne(sessionId);
      filePath =
        session?.earlyPreviewPath || session?.metadata?.earlyPreviewPath;
    } catch (error) {
      // Session not found, will try fallback
    }

    // Fallback to checking artifacts directory directly
    if (!filePath || !existsSync(filePath)) {
      // Check environment variable first (for standalone AppImage)
      const artifactsDir = process.env.ARTIFACTS_DIR;
      const dockerPath = `/app/artifacts/${sessionId}_early_preview.mp4`;
      const localPath = `../.data/artifacts/${sessionId}_early_preview.mp4`; // Go up one level from backend-nestjs to project root
      const envPath = artifactsDir ? path.join(artifactsDir, `${sessionId}_early_preview.mp4`) : null;

      if (envPath && existsSync(envPath)) {
        filePath = envPath;
      } else if (existsSync(dockerPath)) {
        filePath = dockerPath;
      } else if (existsSync(localPath)) {
        filePath = localPath;
      } else {
        throw new NotFoundException('Preview not available');
      }
    }

    const file = createReadStream(filePath);

    res.set({
      'Content-Type': 'video/mp4',
      'Content-Disposition': `inline; filename="preview_${sessionId}.mp4"`,
    });

    return new StreamableFile(file);
  }

  // Helper function to convert SRT to WebVTT format
  private srtToWebVTT(srtContent: string): string {
    // WebVTT header
    let vtt = 'WEBVTT\n\n';

    // Split SRT content into subtitle blocks (double newline separated)
    const blocks = srtContent.trim().split(/\n\s*\n/);

    for (const block of blocks) {
      if (!block.trim()) continue;

      const lines = block
        .trim()
        .split('\n')
        .map((line) => line.trim())
        .filter((line) => line);

      if (lines.length < 3) continue;

      // First line is subtitle number, second is time, rest is text
      const timeLine = lines[1];
      const text = lines.slice(2).join('\n').trim();

      if (!timeLine || !text) continue;

      // Convert SRT time format (HH:MM:SS,mmm --> HH:MM:SS,mmm) to WebVTT (HH:MM:SS.mmm --> HH:MM:SS.mmm)
      // Also ensure proper WebVTT format
      const timeConverted = timeLine.replace(/,/g, '.');

      // Validate time format (should be like "00:00:00.000 --> 00:00:00.000")
      if (!timeConverted.includes('-->')) {
        continue; // Skip invalid time format
      }

      vtt += `${timeConverted}\n${text}\n\n`;
    }

    return vtt;
  }

  // Serve original and translated SRT subtitles for a session (converted to WebVTT)
  @Get('subtitles/:id')
  async getOriginalSubtitles(
    @Param('id') sessionId: string,
    @Res({ passthrough: true }) res: Response,
  ): Promise<StreamableFile> {
    // Prefer session's stored path if available
    let srtPath: string | undefined;
    try {
      const session = this.sessionsService.findOne(sessionId);
      srtPath = session?.result?.originalSrt;
    } catch {}

    // Fallback to artifacts directory
    if (!srtPath || !existsSync(srtPath)) {
      // Check environment variable first (for standalone AppImage)
      const artifactsDir = process.env.ARTIFACTS_DIR;
      const dockerPath = `/app/artifacts/${sessionId}_subtitles.srt`;
      const localPath = path.join(
        process.cwd(),
        '.data',
        'artifacts',
        `${sessionId}_subtitles.srt`,
      );
      const altLocalPath = `../.data/artifacts/${sessionId}_subtitles.srt`;
      const envPath = artifactsDir ? path.join(artifactsDir, `${sessionId}_subtitles.srt`) : null;

      if (envPath && existsSync(envPath)) {
        srtPath = envPath;
      } else if (existsSync(dockerPath)) {
        srtPath = dockerPath;
      } else if (existsSync(localPath)) {
        srtPath = localPath;
      } else if (existsSync(altLocalPath)) {
        srtPath = altLocalPath;
      } else {
        console.error(
          `Original subtitles not found for session ${sessionId}. Checked:`,
          {
            dockerPath,
            localPath,
            altLocalPath,
            sessionSrt: srtPath,
          },
        );
        throw new NotFoundException(
          `Original subtitles not found for session ${sessionId}`,
        );
      }
    }

    // Read SRT file and convert to WebVTT
    const fs = require('fs');
    const srtContent = fs.readFileSync(srtPath, 'utf-8');
    const vttContent = this.srtToWebVTT(srtContent);

    // Create a Buffer from the WebVTT content
    const vttBuffer = Buffer.from(vttContent, 'utf-8');

    res.set({
      'Content-Type': 'text/vtt; charset=utf-8', // WebVTT format for browser compatibility
      'Content-Disposition': `inline; filename="${sessionId}_subtitles.vtt"`,
      'Cache-Control': 'no-cache',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Range',
      'Access-Control-Expose-Headers': 'Content-Disposition, Content-Length',
    });

    // Return WebVTT as StreamableFile
    return new StreamableFile(Buffer.from(vttBuffer));
  }

  @Get('subtitles/:id/translated')
  async getTranslatedSubtitles(
    @Param('id') sessionId: string,
    @Res({ passthrough: true }) res: Response,
  ): Promise<StreamableFile> {
    let srtPath: string | undefined;
    try {
      const session = this.sessionsService.findOne(sessionId);
      srtPath = session?.result?.translatedSrt;
    } catch {}

    if (!srtPath || !existsSync(srtPath)) {
      // Check environment variable first (for standalone AppImage)
      const artifactsDir = process.env.ARTIFACTS_DIR;
      const dockerPath = `/app/artifacts/${sessionId}_translated_subtitles.srt`;
      const localPath = path.join(
        process.cwd(),
        '.data',
        'artifacts',
        `${sessionId}_translated_subtitles.srt`,
      );
      const altLocalPath = `../.data/artifacts/${sessionId}_translated_subtitles.srt`;
      const envPath = artifactsDir ? path.join(artifactsDir, `${sessionId}_translated_subtitles.srt`) : null;

      if (envPath && existsSync(envPath)) {
        srtPath = envPath;
      } else if (existsSync(dockerPath)) {
        srtPath = dockerPath;
      } else if (existsSync(localPath)) {
        srtPath = localPath;
      } else if (existsSync(altLocalPath)) {
        srtPath = altLocalPath;
      } else {
        console.error(
          `Translated subtitles not found for session ${sessionId}. Checked:`,
          {
            dockerPath,
            localPath,
            altLocalPath,
            sessionSrt: srtPath,
          },
        );
        throw new NotFoundException(
          `Translated subtitles not found for session ${sessionId}`,
        );
      }
    }

    // Read SRT file and convert to WebVTT
    const fs = require('fs');
    const srtContent = fs.readFileSync(srtPath, 'utf-8');
    const vttContent = this.srtToWebVTT(srtContent);

    // Create a Buffer from the WebVTT content
    const vttBuffer = Buffer.from(vttContent, 'utf-8');

    res.set({
      'Content-Type': 'text/vtt; charset=utf-8', // WebVTT format for browser compatibility
      'Content-Disposition': `inline; filename="${sessionId}_translated_subtitles.vtt"`,
      'Cache-Control': 'no-cache',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Range',
      'Access-Control-Expose-Headers': 'Content-Disposition, Content-Length',
    });

    // Return WebVTT as StreamableFile
    return new StreamableFile(Buffer.from(vttBuffer));
  }
}
