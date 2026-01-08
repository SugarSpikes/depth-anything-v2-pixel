// ---------------------------------------------------------------------
// Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.quicinc.semanticsegmentation;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.Typeface;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import android.util.AttributeSet;
import android.view.View;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.locks.ReentrantLock;

/**
 * FragmentRender draws the depth prediction image and overlays performance statistics.
 */
public class FragmentRender extends View {
    private final ReentrantLock mLock = new ReentrantLock();
    private Bitmap mBitmap = null;
    private final Rect mTargetRect = new Rect();
    private float fps;
    private long inferTime = 0;
    private long preprocessTime = 0;
    private long postprocessTime = 0;

    // Paint objects
    private final Paint mTextColor = new Paint();
    private final Paint mTextColorSmall = new Paint();
    private final Paint mBackgroundPaint = new Paint();
    private final Paint mAccentPaint = new Paint();

    // Profiling data
    private boolean profilingEnabled = false;
    private final List<Long> inferenceHistory = new ArrayList<>();
    private static final int PROFILE_HISTORY_SIZE = 100;
    private long minInferTime = Long.MAX_VALUE;
    private long maxInferTime = 0;
    private long totalInferTime = 0;
    private int frameCount = 0;

    // Current colormap name
    private String currentColormap = "Turbo";

    // Runtime delegate info
    private String activeDelegate = "Loading...";
    private boolean showRuntimeInfo = false;

    public FragmentRender(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);

        // Main text paint
        mTextColor.setColor(Color.WHITE);
        mTextColor.setTypeface(Typeface.DEFAULT_BOLD);
        mTextColor.setStyle(Paint.Style.FILL);
        mTextColor.setTextSize(48);
        mTextColor.setAntiAlias(true);
        mTextColor.setShadowLayer(4, 2, 2, Color.BLACK);

        // Small text paint
        mTextColorSmall.setColor(Color.parseColor("#B0B0B0"));
        mTextColorSmall.setTypeface(Typeface.DEFAULT);
        mTextColorSmall.setStyle(Paint.Style.FILL);
        mTextColorSmall.setTextSize(36);
        mTextColorSmall.setAntiAlias(true);
        mTextColorSmall.setShadowLayer(3, 1, 1, Color.BLACK);

        // Background paint for stats panel
        mBackgroundPaint.setColor(Color.parseColor("#80000000"));
        mBackgroundPaint.setStyle(Paint.Style.FILL);

        // Accent paint
        mAccentPaint.setColor(Color.parseColor("#03DAC5"));
        mAccentPaint.setTypeface(Typeface.DEFAULT_BOLD);
        mAccentPaint.setStyle(Paint.Style.FILL);
        mAccentPaint.setTextSize(48);
        mAccentPaint.setAntiAlias(true);
        mAccentPaint.setShadowLayer(4, 2, 2, Color.BLACK);
    }

    public void render(Bitmap image, float fps, long inferTime, long preprocessTime, long postprocessTime) {
        this.mBitmap = image;
        this.fps = fps;
        this.inferTime = inferTime;
        this.preprocessTime = preprocessTime;
        this.postprocessTime = postprocessTime;

        // Update profiling stats
        updateProfilingStats(inferTime);

        postInvalidate();
    }

    private void updateProfilingStats(long inferTime) {
        frameCount++;

        // Track min/max/total
        if (inferTime < minInferTime) minInferTime = inferTime;
        if (inferTime > maxInferTime) maxInferTime = inferTime;
        totalInferTime += inferTime;

        // Keep history for detailed analysis
        if (profilingEnabled) {
            inferenceHistory.add(inferTime);
            if (inferenceHistory.size() > PROFILE_HISTORY_SIZE) {
                inferenceHistory.remove(0);
            }
        }
    }

    public void setProfilingEnabled(boolean enabled) {
        this.profilingEnabled = enabled;
        if (!enabled) {
            inferenceHistory.clear();
        }
    }

    public boolean isProfilingEnabled() {
        return profilingEnabled;
    }

    public void resetProfilingStats() {
        minInferTime = Long.MAX_VALUE;
        maxInferTime = 0;
        totalInferTime = 0;
        frameCount = 0;
        inferenceHistory.clear();
    }

    public void setCurrentColormap(String colormap) {
        this.currentColormap = colormap;
    }

    public String getCurrentColormap() {
        return currentColormap;
    }

    public void setActiveDelegate(String delegate) {
        this.activeDelegate = delegate;
    }

    public String getActiveDelegate() {
        return activeDelegate;
    }

    public void setShowRuntimeInfo(boolean show) {
        this.showRuntimeInfo = show;
    }

    public boolean isShowRuntimeInfo() {
        return showRuntimeInfo;
    }

    // Get profiling data for export
    public String getProfilingReport() {
        if (frameCount == 0) return "No data collected";

        float avgMs = (float) totalInferTime / frameCount / 1_000_000f;
        float minMs = (float) minInferTime / 1_000_000f;
        float maxMs = (float) maxInferTime / 1_000_000f;
        float avgFps = 1000f / avgMs;

        StringBuilder report = new StringBuilder();
        report.append("=== Depth Anything V2 Performance Report ===\n\n");
        report.append(String.format("Total Frames: %d\n", frameCount));
        report.append(String.format("Colormap: %s\n", currentColormap));
        report.append(String.format("Backend: %s\n\n", activeDelegate));
        report.append("--- Inference Time ---\n");
        report.append(String.format("  Average: %.1f ms\n", avgMs));
        report.append(String.format("  Min: %.1f ms\n", minMs));
        report.append(String.format("  Max: %.1f ms\n", maxMs));
        report.append(String.format("  Std Dev: %.1f ms\n", calculateStdDev()));
        report.append(String.format("\n--- Throughput ---\n"));
        report.append(String.format("  Average FPS: %.1f\n", avgFps));
        report.append(String.format("  Max FPS: %.1f\n", 1000f / minMs));
        report.append(String.format("  Min FPS: %.1f\n", 1000f / maxMs));

        return report.toString();
    }

    private float calculateStdDev() {
        if (inferenceHistory.size() < 2) return 0;

        long sum = 0;
        for (Long val : inferenceHistory) {
            sum += val;
        }
        float mean = (float) sum / inferenceHistory.size();

        float sumSquaredDiff = 0;
        for (Long val : inferenceHistory) {
            float diff = val - mean;
            sumSquaredDiff += diff * diff;
        }

        return (float) Math.sqrt(sumSquaredDiff / inferenceHistory.size()) / 1_000_000f;
    }

    @SuppressLint("DefaultLocale")
    @Override
    protected void onDraw(@NonNull Canvas canvas) {
        mLock.lock();

        if (mBitmap != null) {
            int insetHeight, insetWidth;

            float canvasRatio = (float) getWidth() / (float) getHeight();
            float bitmapRatio = (float) mBitmap.getWidth() / mBitmap.getHeight();
            if (canvasRatio > bitmapRatio) {
                insetHeight = getHeight();
                insetWidth = (int) ((float) getHeight() * bitmapRatio);
            } else {
                insetWidth = getWidth();
                insetHeight = (int) ((float) getWidth() / bitmapRatio);
            }

            int offsetWidth = (getWidth() - insetWidth) / 2;
            int offsetHeight = (getHeight() - insetHeight) / 2;
            mTargetRect.left = offsetWidth;
            mTargetRect.top = offsetHeight;
            mTargetRect.right = offsetWidth + insetWidth;
            mTargetRect.bottom = offsetHeight + insetHeight;
            canvas.drawBitmap(mBitmap, null, mTargetRect, null);

            // Rotate canvas for text rendering
            canvas.rotate(90, 0, 0);
            canvas.translate(offsetHeight, -insetWidth - offsetWidth);

            // Calculate background height based on what we're showing
            int bgHeight = 320;
            if (showRuntimeInfo) bgHeight += 60;
            if (profilingEnabled) bgHeight += 120;

            // Draw semi-transparent background for stats
            RectF statsBg = new RectF(5, 5, 400, bgHeight);
            canvas.drawRoundRect(statsBg, 15, 15, mBackgroundPaint);

            // Draw stats
            int yPos = 55;
            int lineHeight = 55;

            // FPS with color coding
            Paint fpsPaint = fps >= 5 ? mAccentPaint : (fps >= 2 ? mTextColor : mTextColorSmall);
            canvas.drawText(String.format("FPS: %.1f", fps), 20, yPos, fpsPaint);
            yPos += lineHeight;

            // Colormap indicator
            canvas.drawText("Mode: " + currentColormap, 20, yPos, mTextColorSmall);
            yPos += lineHeight;

            // Runtime delegate info (when enabled)
            if (showRuntimeInfo) {
                canvas.drawText("Backend: " + activeDelegate, 20, yPos, mAccentPaint);
                yPos += lineHeight;
            }

            // Timing breakdown
            canvas.drawText(String.format("Pre: %.0fms", (float) preprocessTime / 1_000_000), 20, yPos, mTextColorSmall);
            yPos += lineHeight;

            canvas.drawText(String.format("Infer: %.0fms", (float) inferTime / 1_000_000), 20, yPos, mTextColor);
            yPos += lineHeight;

            canvas.drawText(String.format("Post: %.0fms", (float) postprocessTime / 1_000_000), 20, yPos, mTextColorSmall);
            yPos += lineHeight;

            // Profiling stats (if enabled)
            if (profilingEnabled && frameCount > 0) {
                yPos += 10; // Extra spacing
                float avgMs = (float) totalInferTime / frameCount / 1_000_000f;
                canvas.drawText(String.format("Avg: %.1fms (%d frames)", avgMs, frameCount), 20, yPos, mAccentPaint);
                yPos += lineHeight;

                canvas.drawText(String.format("Min/Max: %.0f/%.0fms",
                        (float) minInferTime / 1_000_000,
                        (float) maxInferTime / 1_000_000), 20, yPos, mTextColorSmall);
            }

            // App title at bottom
            canvas.drawText("Depth Anything V2", 20, insetWidth - 20, mTextColorSmall);
        }
        mLock.unlock();
    }
}
