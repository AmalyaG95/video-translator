"""
gRPC Translation Service Implementation
Handles translation requests and streams progress back to NestJS API
"""

import asyncio
import logging
from typing import Iterator, Dict, Any
from pathlib import Path
import json

import grpc
import translation_pb2, translation_pb2_grpc
from pipeline.compliant_pipeline import CompliantVideoTranslationPipeline
from utils.path_resolver import path_resolver, get_session_artifacts

logger = logging.getLogger(__name__)


class TranslationServicer(translation_pb2_grpc.TranslationServiceServicer):
    """gRPC servicer for video translation"""
    
    def __init__(self):
        self.pipeline = CompliantVideoTranslationPipeline()
        self.active_sessions: Dict[str, Any] = {}
        self.cancellation_events: Dict[str, asyncio.Event] = {}
        logger.info("Translation servicer initialized")
        
        # Log environment information
        env_info = path_resolver.get_environment_info()
        logger.info(f"Environment: {'Docker' if env_info['is_docker'] else 'Local Development'}")
        logger.info(f"Artifacts directory: {env_info['base_paths']['artifacts']}")
        logger.info(f"Temp work directory: {env_info['base_paths']['temp_work']}")
    
    async def TranslateVideo(
        self,
        request: translation_pb2.TranslationRequest,
        context: grpc.aio.ServicerContext
    ) -> Iterator[translation_pb2.TranslationProgress]:
        """Stream translation progress"""
        session_id = request.session_id
        # FORCE OUTPUT - print always works
        print(f"🌐🌐🌐 TranslateVideo CALLED - session: {session_id}", flush=True)
        print(f"   Request received at gRPC level", flush=True)
        print(f"   File path: {request.file_path}", flush=True)
        print(f"   Languages: {request.source_lang} -> {request.target_lang}", flush=True)
        logger.info(f"🌐 TranslateVideo CALLED - session: {session_id}")
        logger.info(f"   Request received at gRPC level")
        logger.info(f"   File path: {request.file_path}")
        logger.info(f"   Languages: {request.source_lang} -> {request.target_lang}")
        
        # Create cancellation event and progress queue
        cancellation_event = asyncio.Event()
        self.cancellation_events[session_id] = cancellation_event
        progress_queue = asyncio.Queue()
        
        # Sentinel value to signal completion
        DONE = object()
        
        async def run_pipeline():
            """Run pipeline in background task"""
            print(f"🔵🔵🔵 run_pipeline ENTRY - session {session_id}", flush=True)
            print(f"   Request details: file={request.file_path}, source={request.source_lang}, target={request.target_lang}", flush=True)
            logger.info(f"🔵 run_pipeline ENTRY - session {session_id}")
            logger.info(f"   Request details: file={request.file_path}, source={request.source_lang}, target={request.target_lang}")
            try:
                print(f"✅✅✅ run_pipeline try block entered for session {session_id}", flush=True)
                print(f"   Starting pipeline.process_video call...", flush=True)
                logger.info(f"✅ run_pipeline try block entered for session {session_id}")
                logger.info(f"   Starting pipeline.process_video call...")
                
                # Progress callback that puts updates in queue
                async def progress_callback(progress: int, message: str, **kwargs):
                    """Put progress updates in queue"""
                    try:
                        await progress_queue.put({
                            'progress': progress,
                            'current_step': message,
                            'status': kwargs.get('status', 'processing'),
                            'message': kwargs.get('message', ''),
                            'early_preview_available': kwargs.get('early_preview_available', False),
                            'early_preview_path': kwargs.get('early_preview_path', ''),
                            'current_chunk': kwargs.get('current_chunk', 0),
                            'total_chunks': kwargs.get('total_chunks', 0)
                        })
                    except Exception as e:
                        logger.error(f"Error in progress callback: {e}")
                
                # Generate output path using dynamic path resolver
                session_artifacts = get_session_artifacts(session_id)
                output_path = session_artifacts['translated_video']
                
                # Run pipeline with callback
                print(f"📞📞📞 CALLING pipeline.process_video - session {session_id}", flush=True)
                print(f"   video_path: {request.file_path}", flush=True)
                print(f"   output_path: {output_path}", flush=True)
                logger.info(f"📞 CALLING pipeline.process_video - session {session_id}")
                logger.info(f"   video_path: {request.file_path}")
                logger.info(f"   output_path: {output_path}")
                
                result = await self.pipeline.process_video(
                    video_path=Path(request.file_path),
                    source_lang=request.source_lang,
                    target_lang=request.target_lang,
                    output_path=output_path,
                    session_id=session_id,
                    progress_callback=progress_callback
                )
                
                print(f"📥📥📥 pipeline.process_video RETURNED - session {session_id}", flush=True)
                print(f"   Result keys: {list(result.keys()) if result else 'None'}", flush=True)
                print(f"   success: {result.get('success') if result else 'N/A'}", flush=True)
                print(f"   segments_processed: {result.get('segments_processed', 'MISSING') if result else 'N/A'}", flush=True)
                print(f"   error: {result.get('error', 'None') if result else 'N/A'}", flush=True)
                logger.info(f"📥 pipeline.process_video RETURNED - session {session_id}")
                logger.info(f"   Result keys: {list(result.keys()) if result else 'None'}")
                logger.info(f"   success: {result.get('success') if result else 'N/A'}")
                logger.info(f"   segments_processed: {result.get('segments_processed', 'MISSING') if result else 'N/A'}")
                logger.info(f"   error: {result.get('error', 'None') if result else 'N/A'}")
                
                logger.info(f"Pipeline result: success={result.get('success')}, error={result.get('error')}")
                if not result.get('success'):
                    logger.error(f"Pipeline failed: {result.get('error')}")
                    await progress_queue.put({'error': str(result.get('error', 'Unknown error'))})
                
                # Store result with AI insights
                result['ai_insights'] = self.pipeline.get_ai_insights()
                self.active_sessions[session_id] = result
                
                # Signal completion
                await progress_queue.put(DONE)
                
                logger.info(f"Translation completed for session: {session_id}")
                return result
                
            except Exception as e:
                print(f"❌❌❌ CRITICAL ERROR in run_pipeline for session {session_id}: {e}", flush=True)
                print(f"   Exception type: {type(e).__name__}", flush=True)
                print(f"   Exception args: {e.args}", flush=True)
                import traceback
                print(traceback.format_exc(), flush=True)
                logger.error(f"❌ CRITICAL ERROR in run_pipeline for session {session_id}: {e}", exc_info=True)
                logger.error(f"   Exception type: {type(e).__name__}")
                logger.error(f"   Exception args: {e.args}")
                await progress_queue.put({'error': str(e)})
                await progress_queue.put(DONE)
                return {'success': False, 'error': str(e), 'segments_processed': 0}


        # Start pipeline in background
        print(f"🔥🔥🔥 Starting pipeline_task for session {session_id}", flush=True)
        pipeline_task = asyncio.create_task(run_pipeline())
        print(f"   Task created: {pipeline_task}", flush=True)
        print(f"   Task done: {pipeline_task.done()}, cancelled: {pipeline_task.cancelled()}", flush=True)
        
        try:
            # Initial progress
            yield translation_pb2.TranslationProgress(
                session_id=session_id,
                progress=0,
                current_step='Starting translation',
                status='processing',
                message='Initializing pipeline'
            )
            
            # Stream progress updates from queue
            while True:
                try:
                    # Check if context is cancelled
                    if context.cancelled():
                        logger.warning(f"Context cancelled for session: {session_id}")
                        cancellation_event.set()
                        pipeline_task.cancel()
                        break
                    
                    # Get progress update with timeout
                    update = await asyncio.wait_for(progress_queue.get(), timeout=1.0)
                    
                    # Check for completion
                    if update is DONE:
                        break
                    
                    # Check for error
                    if 'error' in update:
                        yield translation_pb2.TranslationProgress(
                            session_id=session_id,
                            status='failed',
                            message=f"Translation failed: {update['error']}"
                        )
                        break
                    
                    # Calculate enhanced progress fields
                    current_progress = update.get('progress', 0)
                    initialization_eta = update.get('initialization_eta_seconds', 0)
                    
                    # Use initialization ETA if we're in initialization phase (0-5% progress)
                    if current_progress <= 5 and initialization_eta > 0:
                        eta_seconds = initialization_eta
                    else:
                        eta_seconds = self.calculate_eta(session_id, current_progress)
                    
                    hardware_info = self.get_hardware_info()
                    available_segments = update.get('available_segments', [])
                    chunks_per_minute = update.get('chunks_per_minute', 0)
                    current_chunk = update.get('current_chunk', 0)
                    total_chunks = update.get('total_chunks', 0)
                    
                    print(f"DEBUG: Sending progress update - current_chunk: {current_chunk}, total_chunks: {total_chunks}")
                    print(f"DEBUG: Full update data: {update}")
                    
                    # Send enhanced progress update
                    yield translation_pb2.TranslationProgress(
                        session_id=session_id,
                        progress=current_progress,
                        current_step=update.get('current_step', ''),
                        status=update.get('status', 'processing'),
                        message=update.get('message', ''),
                        early_preview_available=update.get('early_preview_available', False),
                        early_preview_path=update.get('early_preview_path', ''),
                        # Enhanced fields
                        eta_seconds=eta_seconds,
                        hardware_info=translation_pb2.HardwareInfo(
                            cpu=hardware_info['cpu'],
                            gpu=hardware_info['gpu'],
                            vram_gb=hardware_info['vram_gb'],
                            ram_gb=hardware_info['ram_gb']
                        ),
                        available_segment_ids=available_segments,
                        chunks_per_minute=chunks_per_minute,
                        current_chunk=current_chunk,
                        total_chunks=total_chunks
                    )
                    
                except asyncio.TimeoutError:
                    # Just continue waiting
                    continue
            
            # Send final completion only if no error occurred
            # Check if pipeline task completed successfully
            if pipeline_task.done() and not pipeline_task.cancelled():
                try:
                    pipeline_result = pipeline_task.result()
                    if pipeline_result and pipeline_result.get('success', False):
                        # Extract actual segment counts from pipeline result
                        segments_processed = pipeline_result.get('segments_processed', 0)
                        
                        # CRITICAL: Validate segments were actually processed
                        if segments_processed == 0:
                            logger.error(f"Pipeline marked as success but processed 0 segments for {session_id}")
                            yield translation_pb2.TranslationProgress(
                                session_id=session_id,
                                status='failed',
                                message='Translation failed: No segments were processed'
                            )
                            return
                        
                        # Use segments_processed as both current and total for completion
                        final_current_chunk = segments_processed
                        final_total_chunks = segments_processed
                        
                        logger.info(f"Final completion for {session_id}: {segments_processed} segments processed")
                        
                        yield translation_pb2.TranslationProgress(
                            session_id=session_id,
                            progress=100,
                            current_step='Completed',
                            status='completed',
                            message='Translation completed successfully',
                            current_chunk=final_current_chunk,
                            total_chunks=final_total_chunks
                        )
                    else:
                        # Pipeline returned success=False or None
                        error_msg = pipeline_result.get('error', 'Translation failed during processing') if pipeline_result else 'Translation failed: No result returned'
                        segments_processed = pipeline_result.get('segments_processed', 0) if pipeline_result else 0
                        logger.error(f"Pipeline failed for {session_id}: {error_msg}, segments_processed={segments_processed}")
                        yield translation_pb2.TranslationProgress(
                            session_id=session_id,
                            status='failed',
                            message=error_msg,
                            current_chunk=segments_processed,
                            total_chunks=segments_processed
                        )
                except Exception as e:
                    logger.error(f"Exception getting pipeline result for {session_id}: {e}", exc_info=True)
                    yield translation_pb2.TranslationProgress(
                        session_id=session_id,
                        status='failed',
                        message=f'Translation failed: {str(e)}',
                        current_chunk=0,
                        total_chunks=0
                    )
            else:
                # Pipeline task not done or was cancelled
                if pipeline_task.cancelled():
                    logger.warning(f"Pipeline task was cancelled for {session_id}")
                    message = 'Translation was cancelled'
                else:
                    logger.error(f"Pipeline task not completed for {session_id} (done={pipeline_task.done()}, cancelled={pipeline_task.cancelled()})")
                    message = 'Translation failed: Pipeline did not complete'
                yield translation_pb2.TranslationProgress(
                    session_id=session_id,
                    status='failed',
                    message=message,
                    current_chunk=0,
                    total_chunks=0
                )
            
        except asyncio.CancelledError:
            logger.warning(f"Translation cancelled for session: {session_id}")
            pipeline_task.cancel()
            yield translation_pb2.TranslationProgress(
                session_id=session_id,
                status='failed',
                message='Translation cancelled'
            )
            
        except Exception as e:
            logger.error(f"Streaming error for session {session_id}: {e}", exc_info=True)
            pipeline_task.cancel()
            yield translation_pb2.TranslationProgress(
                session_id=session_id,
                status='failed',
                message=f'Translation failed: {str(e)}'
            )
        
        finally:
            # Cleanup
            self.cancellation_events.pop(session_id, None)
            if not pipeline_task.done():
                pipeline_task.cancel()

    def get_ai_insights(self, session_id: str) -> Dict[str, Any]:
        """Get AI insights for a completed session"""
        session = self.active_sessions.get(session_id)
        if session and 'ai_insights' in session:
            return {
                'insights': session['ai_insights'],
                'total_insights': len(session['ai_insights']),
                'session_id': session_id
            }
        return {
            'insights': [],
            'total_insights': 0,
            'session_id': session_id,
            'error': 'Session not found or no insights available'
        }
    
    async def GetTranslationResult(
        self,
        request: translation_pb2.ResultRequest,
        context: grpc.aio.ServicerContext
    ) -> translation_pb2.TranslationResult:
        """Get translation result"""
        session_id = request.session_id
        logger.info(f"Getting result for session: {session_id}")
        
        result = self.active_sessions.get(session_id)
        
        if not result:
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"No result found for session: {session_id}"
            )
        
        # Extract SRT files from nested dict
        srt_files = result.get('srt_files', {})
        
        # Extract quality metrics if available
        quality_metrics_dict = result.get('quality_metrics', {})
        quality_metrics = None
        if quality_metrics_dict:
            quality_metrics = translation_pb2.QualityMetrics(
                lip_sync_accuracy=quality_metrics_dict.get('lip_sync_accuracy', 85.0),
                voice_quality=quality_metrics_dict.get('voice_quality', 80.0),
                translation_quality=quality_metrics_dict.get('translation_quality', 90.0),
                duration_match=quality_metrics_dict.get('duration_match', True),
                avg_lufs=quality_metrics_dict.get('avg_lufs', -18.0),
                avg_atempo=quality_metrics_dict.get('avg_atempo', 1.0),
                segments_condensed=quality_metrics_dict.get('segments_condensed', 0)
            )
        
        return translation_pb2.TranslationResult(
            session_id=session_id,
            output_path=result.get('output_path', ''),
            status='completed' if result.get('success') else 'failed',
            duration=int(result.get('processing_time', 0)),
            original_srt=srt_files.get('original', ''),
            translated_srt=srt_files.get('translated', ''),
            quality_metrics=quality_metrics,
            output_size=result.get('output_size', 0),
            original_size=result.get('original_size', 0),
            processing_time_seconds=result.get('processing_time_seconds', 0.0)
        )
    
    async def DetectLanguage(
        self,
        request: translation_pb2.LanguageDetectionRequest,
        context: grpc.aio.ServicerContext
    ) -> translation_pb2.LanguageDetectionResult:
        """Detect language from video file"""
        file_path_str = request.file_path
        logger.info(f"Detecting language for file: {file_path_str}")
        
        try:
            # Convert string to Path object (detect_language expects Path)
            from pathlib import Path
            file_path = Path(file_path_str)
            
            # Use the pipeline to detect language
            detected_language, confidence = await self.pipeline.detect_language(file_path)
            
            return translation_pb2.LanguageDetectionResult(
                detected_language=detected_language,
                confidence=confidence,
                success=True,
                message=f"Language detected: {detected_language} (confidence: {confidence:.2f})"
            )
        except Exception as e:
            logger.error(f"Language detection failed: {str(e)}")
            return translation_pb2.LanguageDetectionResult(
                detected_language='en',  # Default to English
                confidence=0.5,
                success=False,
                message=f"Language detection failed: {str(e)}"
            )
    
    async def CancelTranslation(
        self,
        request: translation_pb2.CancelRequest,
        context: grpc.aio.ServicerContext
    ) -> translation_pb2.CancelResponse:
        """Cancel ongoing translation"""
        session_id = request.session_id
        logger.info(f"Cancelling translation for session: {session_id}")
        
        cancellation_event = self.cancellation_events.get(session_id)
        
        if cancellation_event:
            cancellation_event.set()
            return translation_pb2.CancelResponse(
                success=True,
                message=f"Translation cancelled for session: {session_id}"
            )
        else:
            return translation_pb2.CancelResponse(
                success=False,
                message=f"No active translation found for session: {session_id}"
            )

    async def ControlTranslation(
        self,
        request: translation_pb2.ControlTranslationRequest,
        context: grpc.aio.ServicerContext
    ) -> translation_pb2.ControlTranslationResponse:
        """Control translation (pause/resume/cancel)"""
        session_id = request.session_id
        action = request.action
        logger.info(f"Control translation request for session {session_id}: {action}")
        
        try:
            if action == "pause":
                # Set pause flag in active session
                if session_id in self.active_sessions:
                    self.active_sessions[session_id]['paused'] = True
                    logger.info(f"Translation paused for session: {session_id}")
                    return translation_pb2.ControlTranslationResponse(
                        success=True,
                        is_paused=True,
                        message=f"Translation paused for session: {session_id}"
                    )
                else:
                    return translation_pb2.ControlTranslationResponse(
                        success=False,
                        is_paused=False,
                        message=f"No active translation found for session: {session_id}"
                    )
            
            elif action == "resume":
                # Clear pause flag in active session
                if session_id in self.active_sessions:
                    self.active_sessions[session_id]['paused'] = False
                    logger.info(f"Translation resumed for session: {session_id}")
                    return translation_pb2.ControlTranslationResponse(
                        success=True,
                        is_paused=False,
                        message=f"Translation resumed for session: {session_id}"
                    )
                else:
                    return translation_pb2.ControlTranslationResponse(
                        success=False,
                        is_paused=False,
                        message=f"No active translation found for session: {session_id}"
                    )
            
            elif action == "cancel":
                # Cancel translation and export partials
                if session_id in self.cancellation_events:
                    self.cancellation_events[session_id].set()
                    logger.info(f"Translation cancellation requested for session: {session_id}")
                    
                    # Export partial video if possible
                    try:
                        partial_path = await self.export_partial_video(session_id)
                        message = f"Translation cancelled for session: {session_id}"
                        if partial_path:
                            message += f". Partial video exported to: {partial_path}"
                    except Exception as e:
                        logger.warning(f"Failed to export partial video: {e}")
                        message = f"Translation cancelled for session: {session_id}. Partial export failed: {e}"
                    
                    return translation_pb2.ControlTranslationResponse(
                        success=True,
                        is_paused=False,
                        message=message
                    )
                else:
                    return translation_pb2.ControlTranslationResponse(
                        success=False,
                        is_paused=False,
                        message=f"No active translation found for session: {session_id}"
                    )
            
            else:
                return translation_pb2.ControlTranslationResponse(
                    success=False,
                    is_paused=False,
                    message=f"Invalid action: {action}. Must be 'pause', 'resume', or 'cancel'"
                )
                
        except Exception as e:
            logger.error(f"Error controlling translation: {e}")
            return translation_pb2.ControlTranslationResponse(
                success=False,
                is_paused=False,
                message=f"Error controlling translation: {e}"
            )

    async def GetAIInsights(
        self,
        request: translation_pb2.AIInsightsRequest,
        context: grpc.aio.ServicerContext
    ) -> translation_pb2.AIInsightsResponse:
        """Get AI insights for a session"""
        session_id = request.session_id
        logger.info(f"Getting AI insights for session: {session_id}")
        
        try:
            insights_data = self.get_ai_insights(session_id)
            
            # Convert insights to protobuf format
            insights = []
            for insight in insights_data.get('insights', []):
                pb_insight = translation_pb2.AIInsight(
                    id=insight.get('id', ''),
                    type=insight.get('type', ''),
                    title=insight.get('title', ''),
                    description=insight.get('description', ''),
                    impact=insight.get('impact', ''),
                    timestamp=insight.get('timestamp', ''),
                    data=json.dumps(insight.get('data', {}))
                )
                insights.append(pb_insight)
            
            return translation_pb2.AIInsightsResponse(
                session_id=session_id,
                insights=insights,
                total_insights=insights_data.get('total_insights', 0),
                success=True,
                message='AI insights retrieved successfully'
            )
            
        except Exception as e:
            logger.error(f"Error getting AI insights for session {session_id}: {e}")
            return translation_pb2.AIInsightsResponse(
                session_id=session_id,
                insights=[],
                total_insights=0,
                success=False,
                message=f'Error retrieving AI insights: {str(e)}'
            )

    async def export_partial_video(self, session_id: str) -> str | None:
        """Export partial video from completed segments"""
        try:
            # Get session info
            session_info = self.active_sessions.get(session_id, {})
            if not session_info:
                return None
            
            # Get completed segments from pipeline
            pipeline = session_info.get('pipeline')
            if not pipeline:
                return None
            
            # Use pipeline's export_partial method
            partial_path = await pipeline.export_partial_video(session_id)
            return partial_path
            
        except Exception as e:
            logger.error(f"Failed to export partial video for session {session_id}: {e}")
            return None

    def get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information for ETA calculation"""
        try:
            import psutil
            import torch
            
            # Get CPU info
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            cpu_name = f"CPU ({cpu_count} cores)"
            if cpu_freq:
                cpu_name += f" @ {cpu_freq.max:.0f}MHz"
            
            # Get GPU info
            gpu_name = "No GPU"
            vram_gb = 0
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            
            # Get RAM info
            ram_gb = psutil.virtual_memory().total // (1024**3)
            
            return {
                'cpu': cpu_name,
                'gpu': gpu_name,
                'vram_gb': vram_gb,
                'ram_gb': ram_gb
            }
            
        except Exception as e:
            logger.warning(f"Failed to get hardware info: {e}")
            return {
                'cpu': 'Unknown',
                'gpu': 'Unknown',
                'vram_gb': 0,
                'ram_gb': 8
            }

    def calculate_eta(self, session_id: str, current_progress: float) -> int:
        """Calculate estimated time remaining in seconds"""
        try:
            session_info = self.active_sessions.get(session_id, {})
            if not session_info:
                return 0
            
            # Get timing info
            start_time = session_info.get('start_time')
            if not start_time:
                return 0
            
            import time
            elapsed_time = time.time() - start_time
            
            if current_progress <= 0:
                return 0
            
            # Calculate ETA based on current progress
            total_estimated_time = elapsed_time / (current_progress / 100)
            remaining_time = total_estimated_time - elapsed_time
            
            return max(0, int(remaining_time))
            
        except Exception as e:
            logger.warning(f"Failed to calculate ETA: {e}")
            return 0
