// src/database/schema.ts

export const schema = `
-- Arms (Multi-Armed Bandit)
CREATE TABLE IF NOT EXISTS arms (
  id TEXT PRIMARY KEY,
  format TEXT NOT NULL,
  hook_type TEXT NOT NULL,
  caption_style TEXT NOT NULL,
  duration TEXT NOT NULL,
  post_hour INTEGER NOT NULL,
  alpha REAL NOT NULL DEFAULT 1.0,
  beta REAL NOT NULL DEFAULT 1.0,
  pulls INTEGER NOT NULL DEFAULT 0,
  total_reward REAL NOT NULL DEFAULT 0.0,
  features TEXT NOT NULL,
  last_updated TEXT NOT NULL,
  created_at TEXT NOT NULL
);

-- Accounts
CREATE TABLE IF NOT EXISTS accounts (
  id TEXT PRIMARY KEY,
  username TEXT NOT NULL UNIQUE,
  access_token TEXT NOT NULL,
  refresh_token TEXT NOT NULL,
  open_id TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  niche TEXT NOT NULL,
  is_active INTEGER NOT NULL DEFAULT 1,
  daily_post_limit INTEGER NOT NULL DEFAULT 10,
  min_post_interval INTEGER NOT NULL DEFAULT 120,
  created_at TEXT NOT NULL
);

-- Content Ideas
CREATE TABLE IF NOT EXISTS content_ideas (
  id TEXT PRIMARY KEY,
  niche TEXT NOT NULL,
  topic TEXT NOT NULL,
  hook TEXT NOT NULL,
  script TEXT NOT NULL,
  caption TEXT NOT NULL,
  hashtags TEXT NOT NULL,
  format TEXT NOT NULL,
  hook_type TEXT NOT NULL,
  caption_style TEXT NOT NULL,
  duration TEXT NOT NULL,
  viral_score REAL NOT NULL,
  arm_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY (arm_id) REFERENCES arms(id)
);

-- Generated Content
CREATE TABLE IF NOT EXISTS generated_content (
  id TEXT PRIMARY KEY,
  idea_id TEXT NOT NULL,
  audio_path TEXT NOT NULL,
  video_path TEXT NOT NULL,
  thumbnail_path TEXT NOT NULL,
  subtitles_path TEXT,
  duration REAL NOT NULL,
  resolution TEXT NOT NULL,
  fps INTEGER NOT NULL,
  codec TEXT NOT NULL,
  size INTEGER NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY (idea_id) REFERENCES content_ideas(id)
);

-- Posts
CREATE TABLE IF NOT EXISTS posts (
  id TEXT PRIMARY KEY,
  account_id TEXT NOT NULL,
  content_id TEXT NOT NULL,
  tiktok_video_id TEXT,
  posted_at TEXT,
  scheduled_for TEXT,
  is_draft INTEGER NOT NULL DEFAULT 1,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY (account_id) REFERENCES accounts(id),
  FOREIGN KEY (content_id) REFERENCES generated_content(id)
);

-- Metrics
CREATE TABLE IF NOT EXISTS metrics (
  id TEXT PRIMARY KEY,
  post_id TEXT NOT NULL,
  video_id TEXT NOT NULL,
  timestamp TEXT NOT NULL,
  views INTEGER NOT NULL DEFAULT 0,
  likes INTEGER NOT NULL DEFAULT 0,
  comments INTEGER NOT NULL DEFAULT 0,
  shares INTEGER NOT NULL DEFAULT 0,
  saves INTEGER NOT NULL DEFAULT 0,
  completion_rate REAL NOT NULL DEFAULT 0.0,
  average_watch_time REAL NOT NULL DEFAULT 0.0,
  engagement_rate REAL NOT NULL DEFAULT 0.0,
  share_rate REAL NOT NULL DEFAULT 0.0,
  comment_rate REAL NOT NULL DEFAULT 0.0,
  FOREIGN KEY (post_id) REFERENCES posts(id)
);

-- Daily Plans
CREATE TABLE IF NOT EXISTS daily_plans (
  id TEXT PRIMARY KEY,
  date TEXT NOT NULL,
  account_id TEXT NOT NULL,
  content_ideas TEXT NOT NULL,
  schedule TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY (account_id) REFERENCES accounts(id)
);

-- Worker Jobs
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  type TEXT NOT NULL,
  payload TEXT NOT NULL,
  priority INTEGER NOT NULL DEFAULT 0,
  attempts INTEGER NOT NULL DEFAULT 0,
  max_retries INTEGER NOT NULL DEFAULT 3,
  status TEXT NOT NULL,
  error TEXT,
  created_at TEXT NOT NULL,
  started_at TEXT,
  completed_at TEXT
);

-- Model Predictions (for tracking)
CREATE TABLE IF NOT EXISTS predictions (
  id TEXT PRIMARY KEY,
  arm_id TEXT NOT NULL,
  features TEXT NOT NULL,
  probability REAL NOT NULL,
  confidence REAL NOT NULL,
  actual_success INTEGER,
  timestamp TEXT NOT NULL,
  FOREIGN KEY (arm_id) REFERENCES arms(id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_arms_pulls ON arms(pulls);
CREATE INDEX IF NOT EXISTS idx_posts_account ON posts(account_id);
CREATE INDEX IF NOT EXISTS idx_posts_status ON posts(status);
CREATE INDEX IF NOT EXISTS idx_metrics_post ON metrics(post_id);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_type ON jobs(type);
CREATE INDEX IF NOT EXISTS idx_content_arm ON content_ideas(arm_id);
`;
