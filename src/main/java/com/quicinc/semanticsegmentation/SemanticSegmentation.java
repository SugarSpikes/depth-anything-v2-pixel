// ---------------------------------------------------------------------
// Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.quicinc.semanticsegmentation;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;
import android.util.Pair;

import com.quicinc.tflite.AIHubDefaults;
import com.quicinc.tflite.TFLiteHelpers;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.osgi.OpenCVNativeLoader;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class SemanticSegmentation implements AutoCloseable {
    private static final String TAG = "DepthAnythingV2";
    private final Interpreter tfLiteInterpreter;
    private final Map<TFLiteHelpers.DelegateType, Delegate> tfLiteDelegateStore;
    private final int[] inputShape;
    private final int[] outputShape;
    private long preprocessingTime;
    private long inferenceTime;
    private long postprocessingTime;
    private final int outputChannels; // Depth output channels
    private final String activeDelegate; // Which delegate is actually being used

    // Colormap options for depth visualization
    public enum ColormapType {
        TURBO(Imgproc.COLORMAP_TURBO),
        MAGMA(Imgproc.COLORMAP_MAGMA),
        VIRIDIS(Imgproc.COLORMAP_VIRIDIS),
        PLASMA(Imgproc.COLORMAP_PLASMA),
        INFERNO(Imgproc.COLORMAP_INFERNO),
        GRAYSCALE(-1); // Special case: no colormap

        private final int opencvCode;
        ColormapType(int code) { this.opencvCode = code; }
        public int getCode() { return opencvCode; }
    }

    private ColormapType currentColormap = ColormapType.TURBO;
    private float depthOverlayAlpha = 0.7f; // How much depth vs camera (0.0-1.0)

    // Re-usable memory
    private final ByteBuffer inputByteBuffer;
    private final float[] inputFloatArray;
    private final Mat inputMatAbgr;
    private final Mat inputMatBgr;
    private final Mat inputMatRgb;
    private Mat outputDepthMap;

    /**
     * Create a Depth Estimator from the given model.
     * Uses default compute units: NPU, GPU, CPU.
     * Ignores compute units that fail to load.
     *
     * @param context App context.
     * @param modelPath Model path to load.
     * @throws IOException If the model can't be read from disk.
     */
    public SemanticSegmentation(Context context,
                               String modelPath,
                               TFLiteHelpers.DelegateType[][] delegatePriorityOrder) throws IOException, NoSuchAlgorithmException {
        // Initialize OpenCV
        new OpenCVNativeLoader().init();

        // Load TF Lite model
        Pair<MappedByteBuffer, String> modelAndHash = TFLiteHelpers.loadModelFile(context.getAssets(), modelPath);
        Pair<Interpreter, Map<TFLiteHelpers.DelegateType, Delegate>> iResult = TFLiteHelpers.CreateInterpreterAndDelegatesFromOptions(
                modelAndHash.first,
                delegatePriorityOrder,
                AIHubDefaults.numCPUThreads,
                context.getApplicationInfo().nativeLibraryDir,
                context.getCacheDir().getAbsolutePath(),
                modelAndHash.second
        );
        tfLiteInterpreter = iResult.first;
        tfLiteDelegateStore = iResult.second;

        // Determine active delegate for display
        if (tfLiteDelegateStore.containsKey(TFLiteHelpers.DelegateType.QNN_GPU)) {
            activeDelegate = "QNN GPU";
        } else if (tfLiteDelegateStore.containsKey(TFLiteHelpers.DelegateType.QNN_NPU)) {
            activeDelegate = "QNN NPU (HTP)";
        } else if (tfLiteDelegateStore.containsKey(TFLiteHelpers.DelegateType.GPUv2)) {
            activeDelegate = "TFLite GPU";
        } else {
            activeDelegate = "CPU (XNNPack)";
        }

        Log.i(TAG, "Active delegate: " + activeDelegate);
        Log.i(TAG, "Delegates loaded: " + tfLiteDelegateStore.keySet().toString());

        // Validate TF Lite model fits requirements for this app
        Tensor inputTensor = tfLiteInterpreter.getInputTensor(0);
        inputShape = inputTensor.shape();
        DataType inputType = inputTensor.dataType();
        Log.i(TAG, "Input shape: " + Arrays.toString(inputShape) + ", type: " + inputType);

        if (inputShape.length != 4 || inputShape[0] != 1 || inputShape[3] != 3) {
            throw new RuntimeException("Invalid input tensor shape: " + Arrays.toString(inputShape));
        }
        if (inputType != DataType.FLOAT32) {
            throw new RuntimeException("Expected FLOAT32 input, got: " + inputType);
        }

        Tensor outputTensor = tfLiteInterpreter.getOutputTensor(0);
        outputShape = outputTensor.shape();
        DataType outputType = outputTensor.dataType();
        Log.i(TAG, "Output shape: " + Arrays.toString(outputShape) + ", type: " + outputType);

        if (outputShape.length != 4 || outputShape[0] != 1) {
            throw new RuntimeException("Invalid output tensor shape: " + Arrays.toString(outputShape));
        }
        if (outputType != DataType.FLOAT32) {
            throw new RuntimeException("Expected FLOAT32 output, got: " + outputType);
        }

        outputChannels = outputShape[3];

        int inputHeight = inputShape[1];
        int inputWidth = inputShape[2];

        int outputHeight = outputShape[1];
        int outputWidth = outputShape[2];

        // Allocate re-usable memory
        inputByteBuffer = ByteBuffer.allocateDirect(inputHeight * inputWidth * 3 * 4);
        inputByteBuffer.order(ByteOrder.nativeOrder());

        inputFloatArray = new float[inputHeight * inputWidth * 3];

        inputMatAbgr = new Mat(inputWidth, inputHeight, CvType.CV_8UC4);
        inputMatRgb = new Mat(inputWidth, inputHeight, CvType.CV_8UC3);
        inputMatBgr = new Mat(inputWidth, inputHeight, CvType.CV_8UC3);
        outputDepthMap = new Mat(outputWidth, outputHeight, CvType.CV_32FC1);
    }

    /**
     * @return neural network model input width
     */
    public int getInputWidth() {
        return inputShape[2];
    }

    /**
     * @return neural network model input height
     */
    public int getInputHeight() {
        return inputShape[1];
    }

    /**
     * Set the colormap for depth visualization.
     * @param colormap The colormap type to use
     */
    public void setColormap(ColormapType colormap) {
        this.currentColormap = colormap;
        Log.i(TAG, "Colormap changed to: " + colormap.name());
    }

    /**
     * @return current colormap type
     */
    public ColormapType getColormap() {
        return currentColormap;
    }

    /**
     * Set how much the depth overlay should blend with camera feed.
     * @param alpha 0.0 = camera only, 1.0 = depth only
     */
    public void setDepthOverlayAlpha(float alpha) {
        this.depthOverlayAlpha = Math.max(0f, Math.min(1f, alpha));
    }

    /**
     * @return current depth overlay alpha
     */
    public float getDepthOverlayAlpha() {
        return depthOverlayAlpha;
    }

    /**
     * Get all available colormap types.
     * @return array of colormap types
     */
    public static ColormapType[] getAvailableColormaps() {
        return ColormapType.values();
    }

    /**
     * @return the name of the active delegate (QNN GPU, QNN NPU, TFLite GPU, or CPU)
     */
    public String getActiveDelegate() {
        return activeDelegate;
    }

    /**
     * Free resources used by the depth estimator.
     */
    @Override
    public void close() {
        tfLiteInterpreter.close();
        for (Delegate delegate: tfLiteDelegateStore.values()) {
            delegate.close();
        }
    }

    /**
     * @return last preprocessing time in microseconds.
     */
    public long getLastPreprocessingTime() {
        return preprocessingTime;
    }

    /**
     * @return last inference time in microseconds.
     */
    public long getLastInferenceTime() {
        return inferenceTime;
    }

    /**
     * @return last postprocessing time in microseconds.
     */
    public long getLastPostprocessingTime() {
        return postprocessingTime;
    }

    /**
     * Predicts depth map and visualizes it as a grayscale image
     * @param image Input image
     * @param sensorOrientation Sensor orientation in degrees. If input image is rotated this
     *                          number of degrees clockwise, the image should be upright.
     *
     * @return RGB bitmap of same size and orientation as input image, with depth visualization overlay.
     */
    public Bitmap predict(Bitmap image, int sensorOrientation) {
        long preStartTime = System.nanoTime();

        //
        // Preprocessing
        //

        // Copy input image into OpenCV Mat
        Utils.bitmapToMat(image, inputMatAbgr);

        // OpenCV loads the image as BGR, but the network expects RGB
        Imgproc.cvtColor(inputMatAbgr, inputMatRgb, Imgproc.COLOR_BGRA2RGB);
        Imgproc.cvtColor(inputMatAbgr, inputMatBgr, Imgproc.COLOR_BGRA2BGR);

        // Rotate if necessary
        Mat correctRotInputImageRgb = new Mat();
        switch (sensorOrientation) {
            case 0:
                correctRotInputImageRgb = inputMatRgb;
                break;
            case 90:
                Core.rotate(inputMatRgb, correctRotInputImageRgb, Core.ROTATE_90_COUNTERCLOCKWISE);
                break;
            case 180:
                Core.rotate(inputMatRgb, correctRotInputImageRgb, Core.ROTATE_180);
                break;
            case 270:
                Core.rotate(inputMatRgb, correctRotInputImageRgb, Core.ROTATE_90_CLOCKWISE);
                break;
            default:
                break;
        }

        int inputHeight = inputShape[1];
        int inputWidth = inputShape[2];

        // Scale image to the network
        Mat scaledImage = new Mat(inputHeight, inputWidth, CvType.CV_8UC3);
        Imgproc.resize(correctRotInputImageRgb, scaledImage, scaledImage.size(), 0, 0, Imgproc.INTER_LINEAR);
        scaledImage.convertTo(scaledImage, CvType.CV_32FC3, 1 / 255f);

        //
        // TFLite inference
        //

        // Convert from OpenCV to TFLite expected ByteBuffer
        // To minimize IO overhead, we create a direct-allocated buffer in native order.

        scaledImage.get(0, 0, inputFloatArray);

        // CRITICAL: Reset buffer position before writing new frame data
        inputByteBuffer.rewind();
        FloatBuffer inputFloatBuffer = inputByteBuffer.asFloatBuffer();
        inputFloatBuffer.put(inputFloatArray);
        inputByteBuffer.rewind(); // Reset again for TFLite to read from beginning

        long inferenceStartTime = System.nanoTime();
        preprocessingTime = inferenceStartTime - preStartTime;

        // Run inference
        ByteBuffer[] inputs = new ByteBuffer[] {inputByteBuffer};
        tfLiteInterpreter.runForMultipleInputsOutputs(inputs, new HashMap<>());

        //
        // Postprocessing - Depth Map Normalization
        //

        long postStartTime = System.nanoTime();
        inferenceTime = postStartTime - inferenceStartTime;

        int outputHeight = outputShape[1];
        int outputWidth = outputShape[2];

        // Get raw depth output from model
        ByteBuffer outputBuffer = tfLiteInterpreter.getOutputTensor(0).asReadOnlyBuffer();
        FloatBuffer floatBuf = outputBuffer.asFloatBuffer();
        float[] depthArray = new float[floatBuf.remaining()];
        floatBuf.get(depthArray);

        Log.d(TAG, "Depth array size: " + depthArray.length + " (expected: " + (outputHeight * outputWidth * outputChannels) + ")");

        // Find min and max depth values for normalization
        float minDepth = Float.MAX_VALUE;
        float maxDepth = Float.MIN_VALUE;
        for (float depth : depthArray) {
            if (depth < minDepth) minDepth = depth;
            if (depth > maxDepth) maxDepth = depth;
        }

        Log.d(TAG, "Depth range: min=" + minDepth + ", max=" + maxDepth);

        // Normalize depth values to 0-255 range and create grayscale image
        byte[] normalizedDepth = new byte[depthArray.length];
        float depthRange = maxDepth - minDepth;
        if (depthRange < 0.0001f) {
            depthRange = 1.0f; // Avoid division by zero
        }

        for (int i = 0; i < depthArray.length; i++) {
            float normalizedValue = ((depthArray[i] - minDepth) / depthRange) * 255.0f;
            normalizedDepth[i] = (byte) Math.max(0, Math.min(255, (int) normalizedValue));
        }

        // Create grayscale depth map
        Mat depthMat = new Mat(outputHeight, outputWidth, CvType.CV_8UC1);
        depthMat.put(0, 0, normalizedDepth);

        // Rotate output to match input
        switch (sensorOrientation) {
            case 90:
                Core.rotate(depthMat, depthMat, Core.ROTATE_90_CLOCKWISE);
                break;
            case 180:
                Core.rotate(depthMat, depthMat, Core.ROTATE_180);
                break;
            case 270:
                Core.rotate(depthMat, depthMat, Core.ROTATE_90_COUNTERCLOCKWISE);
                break;
            default:
                break;
        }

        // Resize back to input size
        Mat resizedDepth = new Mat(image.getHeight(), image.getWidth(), CvType.CV_8UC1);
        Imgproc.resize(depthMat, resizedDepth, resizedDepth.size(), 0, 0, Imgproc.INTER_LINEAR);

        // Apply selected colormap for depth visualization
        Mat depthColor = new Mat(image.getHeight(), image.getWidth(), CvType.CV_8UC3);
        if (currentColormap == ColormapType.GRAYSCALE) {
            // For grayscale, just convert to BGR without colormap
            Imgproc.cvtColor(resizedDepth, depthColor, Imgproc.COLOR_GRAY2BGR);
        } else {
            // Apply the selected colormap
            Imgproc.applyColorMap(resizedDepth, depthColor, currentColormap.getCode());
        }

        // Overlay depth map on top of camera preview with configurable alpha
        Core.addWeighted(inputMatBgr, 1.0 - depthOverlayAlpha, depthColor, depthOverlayAlpha, 0.0, inputMatBgr);

        // Convert from OpenCV Mat to Android Bitmap
        Bitmap outputBitmap = Bitmap.createBitmap(inputMatBgr.width(), inputMatBgr.height(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(inputMatBgr, outputBitmap);
        long endTime = System.nanoTime();
        postprocessingTime = endTime - postStartTime;

        Log.d(TAG, String.format("Timing - Pre: %.1fms, Inference: %.1fms, Post: %.1fms",
            preprocessingTime/1e6, inferenceTime/1e6, postprocessingTime/1e6));

        return outputBitmap;
    }
}
