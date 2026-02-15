// src/config/index.ts

import dotenv from 'dotenv';
import { AppConfig } from '../types';

dotenv.config();

const requiredEnvVars = [
  'TIKTOK_CLIENT_KEY',
  'TIKTOK_CLIENT_SECRET',
  'OLLAMA_URL',
  'PIPER_MODEL_PATH',
  'WHISPER_MODEL_PATH'
];

// Validate environment variables
for (const varName of requiredEnvVars) {
  if (!process.env[varName]) {
    throw new Error(`Missing required environment variable: ${varName}`);
  }
}

export const config: AppConfig = {
  env: (process.env.NODE_ENV as 'development' | 'production') || 'development',

  server: {
    strategyPort: parseInt(process.env.STRATEGY_PORT || '3001'),
    coordinatorPort: parseInt(process.env.COORDINATOR_PORT || '3002'),
    workerCount: parseInt(process.env.WORKER_COUNT || '3'),
  },

  ollama: {
    url: process.env.OLLAMA_URL || 'http://localhost:11434',
    model: process.env.OLLAMA_MODEL || 'llama3.2',
    temperature: parseFloat(process.env.OLLAMA_TEMPERATURE || '0.7'),
    maxTokens: parseInt(process.env.OLLAMA_MAX_TOKENS || '2000'),
  },

  piper: {
    modelPath: process.env.PIPER_MODEL_PATH!,
    voice: process.env.PIPER_VOICE || 'en_US-lessac-medium',
    speed: parseFloat(process.env.PIPER_SPEED || '1.0'),
  },

  whisper: {
    modelPath: process.env.WHISPER_MODEL_PATH!,
    language: process.env.WHISPER_LANGUAGE || 'en',
  },

  ffmpeg: {
    resolution: process.env.FFMPEG_RESOLUTION || '1080x1920',
    fps: parseInt(process.env.FFMPEG_FPS || '30'),
    codec: process.env.FFMPEG_CODEC || 'libx264',
    audioCodec: process.env.FFMPEG_AUDIO_CODEC || 'aac',
    bitrate: process.env.FFMPEG_BITRATE || '3000k',
  },

  tiktok: {
    clientKey: process.env.TIKTOK_CLIENT_KEY!,
    clientSecret: process.env.TIKTOK_CLIENT_SECRET!,
    redirectUri: process.env.TIKTOK_REDIRECT_URI || 'http://localhost:3000/callback',
    apiVersion: process.env.TIKTOK_API_VERSION || 'v2',
    maxDailyPosts: parseInt(process.env.MAX_DAILY_POSTS || '10'),
    minPostInterval: parseInt(process.env.MIN_POST_INTERVAL || '120'), // dakika
  },

  strategy: {
    candidateCount: parseInt(process.env.CANDIDATE_COUNT || '10'),
    selectionCount: parseInt(process.env.SELECTION_COUNT || '3'),
    explorationRate: parseFloat(process.env.EXPLORATION_RATE || '0.1'),
    successThreshold: {
      completionRate: parseFloat(process.env.SUCCESS_COMPLETION_RATE || '0.7'),
      engagementRate: parseFloat(process.env.SUCCESS_ENGAGEMENT_RATE || '0.08'),
      viewsMultiplier: parseFloat(process.env.SUCCESS_VIEWS_MULTIPLIER || '1.8'),
    },
    rewardWeights: {
      completion: parseFloat(process.env.REWARD_WEIGHT_COMPLETION || '0.4'),
      engagement: parseFloat(process.env.REWARD_WEIGHT_ENGAGEMENT || '0.25'),
      share: parseFloat(process.env.REWARD_WEIGHT_SHARE || '0.2'),
      comment: parseFloat(process.env.REWARD_WEIGHT_COMMENT || '0.15'),
    },
  },

  database: {
    path: process.env.DB_PATH || './data/tiktok.db',
    backup: process.env.DB_BACKUP === 'true',
    backupInterval: parseInt(process.env.DB_BACKUP_INTERVAL || '86400'), // saniye
  },

  logging: {
    level: (process.env.LOG_LEVEL as any) || 'info',
    dir: process.env.LOG_DIR || './logs',
    maxFiles: parseInt(process.env.LOG_MAX_FILES || '14'),
    maxSize: parseInt(process.env.LOG_MAX_SIZE || '10485760'), // 10MB
  },
};

// Validation
if (config.strategy.candidateCount < config.strategy.selectionCount) {
  throw new Error('CANDIDATE_COUNT must be >= SELECTION_COUNT');
}

if (config.server.workerCount < 1) {
  throw new Error('WORKER_COUNT must be >= 1');
}

const totalWeight =
  config.strategy.rewardWeights.completion +
  config.strategy.rewardWeights.engagement +
  config.strategy.rewardWeights.share +
  config.strategy.rewardWeights.comment;

if (Math.abs(totalWeight - 1.0) > 0.01) {
  throw new Error('Reward weights must sum to 1.0');
}

export default config;
