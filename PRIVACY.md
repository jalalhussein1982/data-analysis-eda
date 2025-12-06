# Privacy Policy - Data Preparation Pipeline

**Effective Date:** December 2025
**Last Updated:** December 2025

## What Data We Process

When you use this application, we process:

1. **Uploaded Files:** The data file(s) you upload for analysis (CSV, Excel, Parquet, JSON)
2. **Configuration Choices:** Your selections for data cleaning, transformation, and pipeline settings

## How We Process Your Data

### In-Memory Only

All data processing occurs **exclusively in your browser's memory** (RAM). No data is:
- Saved to our servers
- Logged to files
- Transmitted to third parties
- Used to train machine learning models

### Session-Based Storage

Temporary data states are stored in your browser session only. When you:
- Close the browser tab → All data is deleted
- Click "Delete All Data" → All data is deleted immediately
- Session times out (after ~15 minutes of inactivity) → All data is deleted

### No Third-Party Access

This application does **not**:
- Share your data with analytics providers
- Use third-party cookies
- Transmit your data outside your browser environment

## Your Data Rights (GDPR)

Under GDPR, you have the right to:

- **Access:** View all data being processed (visible in the UI)
- **Rectification:** Modify data during the pipeline
- **Erasure:** Delete all data via "Delete All Data" button
- **Portability:** Export your processed data at any phase
- **Object:** Stop processing by closing the application

## Data Retention

**Retention Period:** Session duration only (maximum ~15 minutes of inactivity)

**Deletion Method:**
- Automatic on session end
- Manual via "Delete All Data" button
- Temporary files purged on application restart

## Technical Safeguards

| Safeguard | Implementation |
|-----------|----------------|
| Encryption | HTTPS for data transmission (when deployed) |
| Access Control | No authentication required; no user accounts |
| Audit Logging | No logs created |
| Data Minimization | Only process uploaded file, no extraneous collection |

## Hosting & Data Location

When deployed on Streamlit Cloud, servers may be located in:
- United States (primary)
- European Union (if deployed on EU servers)

**Important:** Even when hosted externally, **no data is stored on servers**. Processing occurs in-memory during your session only.

## Changes to This Policy

We may update this policy periodically. The "Last Updated" date will reflect the most recent revision.
