import {
  Controller,
  Post,
  Get,
  Delete,
  Param,
  HttpCode,
  HttpStatus,
  NotFoundException,
  StreamableFile,
  Res,
  Body,
  Query,
} from '@nestjs/common';
import type { Response } from 'express';
import { createReadStream, existsSync, statSync } from 'fs';
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
    await this.videoService.startTranslation(sessionId);
    return {
      message: 'Translation started',
      sessionId,
    };
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
      // Try both Docker and local paths
      const dockerPath = `/app/artifacts/${sessionId}_translated.mp4`;
      const localPath = `../.data/artifacts/${sessionId}_translated.mp4`; // Go up one level from backend-nestjs to project root .data folder

      if (existsSync(dockerPath)) {
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

  @Get('download/:id')
  async downloadVideo(
    @Param('id') sessionId: string,
    @Res({ passthrough: true }) res: Response,
  ): Promise<StreamableFile> {
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
      // Try both Docker and local paths
      const dockerPath = `/app/artifacts/${sessionId}_translated.mp4`;
      const localPath = `../.data/artifacts/${sessionId}_translated.mp4`; // Go up one level from backend-nestjs to project root .data folder

      if (existsSync(dockerPath)) {
        filePath = dockerPath;
      } else if (existsSync(localPath)) {
        filePath = localPath;
      } else {
        throw new NotFoundException('Video file not found');
      }
    }

    // Check if file exists and get stats
    if (!existsSync(filePath)) {
      throw new NotFoundException('Video file not found');
    }

    const stats = statSync(filePath);
    const fileSize = stats.size;

    // Set proper headers for video download (and CORS exposure)
    res.set({
      'Content-Type': 'video/mp4',
      'Content-Disposition': `attachment; filename="translated_${sessionId}.mp4"`,
      'Content-Length': fileSize.toString(),
      'Accept-Ranges': 'bytes',
      'Cache-Control': 'no-cache',
      'Access-Control-Expose-Headers': 'Content-Disposition, Content-Length',
    });

    // Stream file using Nest's StreamableFile to avoid connection issues
    const file = createReadStream(filePath);
    return new StreamableFile(file);
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
      // Try both Docker and local paths
      const dockerPath = `/app/artifacts/${sessionId}_early_preview.mp4`;
      const localPath = `../.data/artifacts/${sessionId}_early_preview.mp4`; // Go up one level from backend-nestjs to project root

      if (existsSync(dockerPath)) {
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
      
      const lines = block.trim().split('\n').map(line => line.trim()).filter(line => line);
      
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
      const dockerPath = `/app/artifacts/${sessionId}_subtitles.srt`;
      const localPath = `../.data/artifacts/${sessionId}_subtitles.srt`;
      if (existsSync(dockerPath)) srtPath = dockerPath;
      else if (existsSync(localPath)) srtPath = localPath;
      else throw new NotFoundException('Original subtitles not found');
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
      const dockerPath = `/app/artifacts/${sessionId}_translated_subtitles.srt`;
      const localPath = `../.data/artifacts/${sessionId}_translated_subtitles.srt`;
      if (existsSync(dockerPath)) srtPath = dockerPath;
      else if (existsSync(localPath)) srtPath = localPath;
      else throw new NotFoundException('Translated subtitles not found');
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
