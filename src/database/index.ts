// src/database/index.ts

import Database from 'better-sqlite3';
import { schema } from './schema';
import config from '../config';
import { logger } from '../utils/logger';
import fs from 'fs';
import path from 'path';

class DatabaseService {
  private db: Database.Database;

  constructor() {
    // Dizin oluştur
    const dbDir = path.dirname(config.database.path);
    if (!fs.existsSync(dbDir)) {
      fs.mkdirSync(dbDir, { recursive: true });
    }

    this.db = new Database(config.database.path, {
      verbose: config.env === 'development' ? logger.debug.bind(logger) : undefined,
    });

    // WAL mode for better concurrency
    this.db.pragma('journal_mode = WAL');
    this.db.pragma('synchronous = NORMAL');
    this.db.pragma('foreign_keys = ON');

    this.initialize();
  }

  private initialize() {
    // Run schema
    this.db.exec(schema);
    logger.info('Database initialized');

    // Backup setup
    if (config.database.backup) {
      this.setupBackup();
    }
  }

  private setupBackup() {
    setInterval(() => {
      try {
        const backupPath = `${config.database.path}.backup-${Date.now()}`;
        this.db.backup(backupPath);

        // Keep only last 7 backups
        const backups = fs.readdirSync(path.dirname(config.database.path))
          .filter(f => f.includes('.backup-'))
          .sort()
          .reverse();

        backups.slice(7).forEach(file => {
          fs.unlinkSync(path.join(path.dirname(config.database.path), file));
        });

        logger.info('Database backup created');
      } catch (error) {
        logger.error('Backup failed', { error });
      }
    }, config.database.backupInterval * 1000);
  }

  getDb(): Database.Database {
    return this.db;
  }

  close() {
    this.db.close();
  }
}

export const db = new DatabaseService();
export default db;
