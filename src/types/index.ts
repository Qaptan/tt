// src/types/index.ts

// ============================================
// ARM (Multi-Armed Bandit)
// ============================================

export interface Arm {
  id: string;
  format: ContentFormat;
  hookType: HookType;
  captionStyle: CaptionStyle;
  duration: DurationRange;
  postHour: number;
  alpha: number;  // Beta dağılımı - başarı
  beta: number;   // Beta dağılımı - başarısızlık
  pulls: number;  // Kaç kez seçildi
  totalReward: number;
  lastUpdated: Date;
  features: number[];  // ML için feature vector
}

export type ContentFormat =
  | 'talking-head'
  | 'b-roll'
  | 'text-overlay'
  | 'slideshow'
  | 'animation'
  | 'green-screen';

export type HookType =
  | 'question'
  | 'shocking-stat'
  | 'story-start'
  | 'controversy'
  | 'tutorial'
  | 'list';

export type CaptionStyle =
  | 'emoji-heavy'
  | 'minimal'
  | 'storytelling'
  | 'call-to-action'
  | 'question-based';

export type DurationRange = '7-15s' | '15-30s' | '30-60s';

// ============================================
// CONTENT
// ============================================

export interface ContentIdea {
  id: string;
  niche: string;
  topic: string;
  hook: string;
  script: string;
  caption: string;
  hashtags: string[];
  format: ContentFormat;
  hookType: HookType;
  captionStyle: CaptionStyle;
  duration: DurationRange;
  viralScore: number;
  armId: string;
  createdAt: Date;
}

export interface GeneratedContent {
  id: string;
  ideaId: string;
  audioPath: string;
  videoPath: string;
  thumbnailPath: string;
  subtitlesPath?: string;
  metadata: ContentMetadata;
  status: ContentStatus;
  createdAt: Date;
}

export interface ContentMetadata {
  duration: number;
  resolution: string;
  fps: number;
  codec: string;
  size: number;
}

export type ContentStatus =
  | 'generating'
  | 'ready'
  | 'uploading'
  | 'posted'
  | 'failed';

// ============================================
// TIKTOK
// ============================================

export interface TikTokAccount {
  id: string;
  username: string;
  accessToken: string;
  refreshToken: string;
  openId: string;
  expiresAt: Date;
  niche: string;
  isActive: boolean;
  dailyPostLimit: number;
  minPostInterval: number;  // dakika
  createdAt: Date;
}

export interface TikTokPost {
  id: string;
  accountId: string;
  contentId: string;
  tiktokVideoId: string;
  postedAt: Date;
  scheduledFor?: Date;
  isDraft: boolean;
  status: PostStatus;
}

export type PostStatus =
  | 'scheduled'
  | 'posting'
  | 'posted'
  | 'failed';

export interface TikTokMetrics {
  id: string;
  postId: string;
  videoId: string;
  timestamp: Date;
  views: number;
  likes: number;
  comments: number;
  shares: number;
  saves: number;
  completionRate: number;
  averageWatchTime: number;
  engagementRate: number;
  shareRate: number;
  commentRate: number;
}

// ============================================
// STRATEGY
// ============================================

export interface StrategyDecision {
  selectedArms: Arm[];
  exploration: boolean;
  reasoning: string;
  timestamp: Date;
}

export interface RewardInput {
  armId: string;
  postId: string;
  metrics: TikTokMetrics;
  accountMedian: number;
}

export interface RewardOutput {
  armId: string;
  reward: number;
  isSuccess: boolean;
  components: {
    completionScore: number;
    engagementScore: number;
    shareScore: number;
    commentScore: number;
  };
}

export interface ModelPrediction {
  armId: string;
  probability: number;
  confidence: number;
}

// ============================================
// COORDINATOR
// ============================================

export interface DailyPlan {
  id: string;
  date: Date;
  accountId: string;
  contentIdeas: ContentIdea[];
  schedule: ScheduleEntry[];
  status: PlanStatus;
}

export interface ScheduleEntry {
  contentId: string;
  scheduledTime: Date;
  armId: string;
  priority: number;
}

export type PlanStatus =
  | 'pending'
  | 'generating'
  | 'ready'
  | 'executing'
  | 'completed'
  | 'failed';

export interface WorkerJob {
  id: string;
  type: JobType;
  payload: any;
  priority: number;
  attempts: number;
  maxRetries: number;
  status: JobStatus;
  createdAt: Date;
  startedAt?: Date;
  completedAt?: Date;
  error?: string;
}

export type JobType =
  | 'generate_content'
  | 'render_video'
  | 'upload_video'
  | 'fetch_metrics'
  | 'update_strategy';

export type JobStatus =
  | 'pending'
  | 'running'
  | 'completed'
  | 'failed'
  | 'retrying';

// ============================================
// CONFIG
// ============================================

export interface AppConfig {
  env: 'development' | 'production';
  server: ServerConfig;
  ollama: OllamaConfig;
  piper: PiperConfig;
  whisper: WhisperConfig;
  ffmpeg: FFmpegConfig;
  tiktok: TikTokConfig;
  strategy: StrategyConfig;
  database: DatabaseConfig;
  logging: LoggingConfig;
}

export interface ServerConfig {
  strategyPort: number;
  coordinatorPort: number;
  workerCount: number;
}

export interface OllamaConfig {
  url: string;
  model: string;
  temperature: number;
  maxTokens: number;
}

export interface PiperConfig {
  modelPath: string;
  voice: string;
  speed: number;
}

export interface WhisperConfig {
  modelPath: string;
  language: string;
}

export interface FFmpegConfig {
  resolution: string;
  fps: number;
  codec: string;
  audioCodec: string;
  bitrate: string;
}

export interface TikTokConfig {
  clientKey: string;
  clientSecret: string;
  redirectUri: string;
  apiVersion: string;
  maxDailyPosts: number;
  minPostInterval: number;
}

export interface StrategyConfig {
  candidateCount: number;
  selectionCount: number;
  explorationRate: number;
  successThreshold: {
    completionRate: number;
    engagementRate: number;
    viewsMultiplier: number;
  };
  rewardWeights: {
    completion: number;
    engagement: number;
    share: number;
    comment: number;
  };
}

export interface DatabaseConfig {
  path: string;
  backup: boolean;
  backupInterval: number;
}

export interface LoggingConfig {
  level: 'debug' | 'info' | 'warn' | 'error';
  dir: string;
  maxFiles: number;
  maxSize: number;
}

// ============================================
// UTILITIES
// ============================================

export interface RetryOptions {
  maxAttempts: number;
  delayMs: number;
  exponentialBackoff: boolean;
  onRetry?: (attempt: number, error: Error) => void;
}

export interface RateLimitConfig {
  maxRequests: number;
  windowMs: number;
}

export class AppError extends Error {
  constructor(
    public message: string,
    public code: string,
    public statusCode: number = 500,
    public isOperational: boolean = true
  ) {
    super(message);
    Object.setPrototypeOf(this, AppError.prototype);
    Error.captureStackTrace(this, this.constructor);
  }
}

// ============================================
// STATISTICS
// ============================================

export interface BetaDistribution {
  alpha: number;
  beta: number;
}

export interface StatisticalTest {
  pValue: number;
  significant: boolean;
  confidenceInterval: [number, number];
}
