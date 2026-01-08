// ---------------------------------------------------------------------
// Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.quicinc.semanticsegmentation;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.app.Dialog;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.ColorStateList;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.util.Size;
import android.graphics.SurfaceTexture;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.core.content.ContextCompat;
import androidx.annotation.NonNull;
import androidx.fragment.app.DialogFragment;
import androidx.fragment.app.Fragment;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureFailure;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CaptureResult;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.view.LayoutInflater;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.Toast;
import org.opencv.android.OpenCVLoader;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.Locale;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

public class CameraFragment extends Fragment {
    private long lastTic = 0;

    private FragmentRender mFragmentRender;
    private float fps = 0;
    private String mCameraId;
    private static final String FRAGMENT_DIALOG = "dialog";
    private SemanticSegmentation segmentor;
    private volatile boolean isInferenceRunning = false;
    private volatile boolean isShowingGalleryResult = false;
    private Bitmap lastProcessedBitmap = null;

    // UI Elements
    private TextureView mTextureView;
    private Button btnTurbo, btnMagma, btnViridis, btnPlasma, btnInferno, btnGrayscale;
    private Button btnGallery, btnRuntime, btnProfile, btnShare;
    private Map<Button, SemanticSegmentation.ColormapType> colormapButtons = new HashMap<>();
    private Button currentSelectedColormap;

    // Gallery picker
    private ActivityResultLauncher<Intent> galleryLauncher;

    // Separate executor for gallery processing (mBackgroundHandler is null when gallery opens)
    private final ExecutorService galleryExecutor = Executors.newSingleThreadExecutor();

    /**
     * {@link TextureView.SurfaceTextureListener} handles several lifecycle events on a
     * {@link TextureView}.
     */
    private final TextureView.SurfaceTextureListener mSurfaceTextureListener
            = new TextureView.SurfaceTextureListener() {

        @Override
        public void onSurfaceTextureAvailable(@NonNull SurfaceTexture texture, int width, int height) {
            openCamera();
        }

        @Override
        public void onSurfaceTextureSizeChanged(@NonNull SurfaceTexture texture, int width, int height) {
        }

        @Override
        public boolean onSurfaceTextureDestroyed(@NonNull SurfaceTexture texture) {
            return true;
        }

        @Override
        public void onSurfaceTextureUpdated(@NonNull SurfaceTexture texture) {
        }

    };

    /**
     * A {@link CameraCaptureSession } for camera preview.
     */
    private CameraCaptureSession mCaptureSession;

    /**
     * A reference to the opened {@link CameraDevice}.
     */
    private CameraDevice mCameraDevice;

    /**
     * The {@link android.util.Size} of camera preview.
     */
    private Size mPreviewSize;

    /**
     * {@link CameraDevice.StateCallback} is called when {@link CameraDevice} changes its state.
     */
    private final CameraDevice.StateCallback mStateCallback = new CameraDevice.StateCallback() {

        @Override
        public void onOpened(@NonNull CameraDevice cameraDevice) {
            mCameraOpenCloseLock.release();
            mCameraDevice = cameraDevice;
            createCameraPreviewSession();
        }

        @Override
        public void onDisconnected(@NonNull CameraDevice cameraDevice) {
            mCameraOpenCloseLock.release();
            cameraDevice.close();
            mCameraDevice = null;
        }

        @Override
        public void onError(@NonNull CameraDevice cameraDevice, int error) {
            mCameraOpenCloseLock.release();
            cameraDevice.close();
            mCameraDevice = null;
            Activity activity = getActivity();
            if (null != activity) {
                activity.finish();
            }
        }

    };

    public static CameraFragment create(SemanticSegmentation segmentor) {
        final CameraFragment fragment = new CameraFragment();
        fragment.segmentor = segmentor;
        return fragment;
    }

    /**
     * An additional thread for running tasks that shouldn't block the UI.
     */
    private HandlerThread mBackgroundThread;

    /**
     * A {@link Handler} for running tasks in the background.
     */
    private Handler mBackgroundHandler;


    /**
     * {@link CaptureRequest.Builder} for the camera preview
     */
    private CaptureRequest.Builder mPreviewRequestBuilder;

    /**
     * A {@link Semaphore} to prevent the app from exiting before closing the camera.
     */
    private final Semaphore mCameraOpenCloseLock = new Semaphore(1);

    /**
     * Orientation of the camera sensor in degrees
     */
    private int mSensorOrientation;

    /**
     * Given {@code choices} of {@code Size}s supported by a camera, choose the smallest one that
     * is at least as large as the minimum width/height.
     */
    private static Size chooseOptimalSize(Size[] choices, int minWidth, int minHeight) {
        List<Size> bigEnough = new ArrayList<>();
        for (Size option : choices) {
            if (option.getWidth() >= minWidth && option.getHeight() >= minHeight) {
                bigEnough.add(option);
            }
        }

        if (bigEnough.size() > 0) {
            return Collections.min(bigEnough, new CompareSizesByArea());
        } else {
            throw new RuntimeException("No suitable camera size found");
        }
    }

    private ActivityResultLauncher<String> requestPermissionLauncher;

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        OpenCVLoader.initDebug();

        requestPermissionLauncher = registerForActivityResult(
                new ActivityResultContracts.RequestPermission(),
                isGranted -> {
                    if (!isGranted) {
                        ErrorDialog.newInstance(getString(R.string.request_permission))
                                .show(getChildFragmentManager(), FRAGMENT_DIALOG);
                    }
                }
        );

        // Gallery image picker
        galleryLauncher = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                result -> {
                    if (result.getResultCode() == Activity.RESULT_OK && result.getData() != null) {
                        Uri imageUri = result.getData().getData();
                        processGalleryImage(imageUri);
                    }
                }
        );

        return inflater.inflate(R.layout.fragment_camera, container, false);
    }

    @Override
    public void onViewCreated(@NonNull final View view, Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        // Initialize views
        mTextureView = view.findViewById(R.id.surface);
        mTextureView.setSurfaceTextureListener(mSurfaceTextureListener);
        mFragmentRender = view.findViewById(R.id.fragmentRender);

        // Initialize colormap buttons
        btnTurbo = view.findViewById(R.id.btnTurbo);
        btnMagma = view.findViewById(R.id.btnMagma);
        btnViridis = view.findViewById(R.id.btnViridis);
        btnPlasma = view.findViewById(R.id.btnPlasma);
        btnInferno = view.findViewById(R.id.btnInferno);
        btnGrayscale = view.findViewById(R.id.btnGrayscale);

        // Map buttons to colormap types
        colormapButtons.put(btnTurbo, SemanticSegmentation.ColormapType.TURBO);
        colormapButtons.put(btnMagma, SemanticSegmentation.ColormapType.MAGMA);
        colormapButtons.put(btnViridis, SemanticSegmentation.ColormapType.VIRIDIS);
        colormapButtons.put(btnPlasma, SemanticSegmentation.ColormapType.PLASMA);
        colormapButtons.put(btnInferno, SemanticSegmentation.ColormapType.INFERNO);
        colormapButtons.put(btnGrayscale, SemanticSegmentation.ColormapType.GRAYSCALE);

        // Set click listeners for colormap buttons
        for (Map.Entry<Button, SemanticSegmentation.ColormapType> entry : colormapButtons.entrySet()) {
            entry.getKey().setOnClickListener(v -> selectColormap(entry.getKey(), entry.getValue()));
        }
        currentSelectedColormap = btnTurbo;

        // Initialize action buttons
        btnGallery = view.findViewById(R.id.btnGallery);
        btnRuntime = view.findViewById(R.id.btnRuntime);
        btnProfile = view.findViewById(R.id.btnProfile);
        btnShare = view.findViewById(R.id.btnShare);

        btnGallery.setOnClickListener(v -> openGallery());
        btnRuntime.setOnClickListener(v -> toggleRuntimeInfo());
        btnProfile.setOnClickListener(v -> toggleProfiling());
        btnShare.setOnClickListener(v -> shareReport());

        // Pass delegate info to renderer
        if (segmentor != null) {
            mFragmentRender.setActiveDelegate(segmentor.getActiveDelegate());
        }

        // Tap on screen to resume camera when showing gallery result
        mFragmentRender.setOnClickListener(v -> {
            if (isShowingGalleryResult) {
                resumeCamera();
                Toast.makeText(getContext(), "Resumed camera", Toast.LENGTH_SHORT).show();
            }
        });
    }

    private void selectColormap(Button selectedButton, SemanticSegmentation.ColormapType colormap) {
        // Update button appearance
        if (currentSelectedColormap != null) {
            currentSelectedColormap.setBackgroundTintList(ColorStateList.valueOf(0xFF444444));
        }
        selectedButton.setBackgroundTintList(ColorStateList.valueOf(
                ContextCompat.getColor(requireContext(), R.color.colorPrimary)));
        currentSelectedColormap = selectedButton;

        // Apply colormap
        if (segmentor != null) {
            segmentor.setColormap(colormap);
            mFragmentRender.setCurrentColormap(colormap.name());
            Toast.makeText(getContext(), "Colormap: " + colormap.name(), Toast.LENGTH_SHORT).show();
        }
    }

    private void openGallery() {
        // If showing gallery result, resume camera first
        if (isShowingGalleryResult) {
            resumeCamera();
        }

        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setType("image/*");
        galleryLauncher.launch(intent);
    }

    private void toggleRuntimeInfo() {
        boolean newState = !mFragmentRender.isShowRuntimeInfo();
        mFragmentRender.setShowRuntimeInfo(newState);

        if (newState) {
            btnRuntime.setText("Hide RT");
            btnRuntime.setBackgroundTintList(ColorStateList.valueOf(0xFF00BCD4)); // Cyan when active
            Toast.makeText(getContext(), "Runtime info: " + mFragmentRender.getActiveDelegate(), Toast.LENGTH_SHORT).show();
        } else {
            btnRuntime.setText("Runtime");
            btnRuntime.setBackgroundTintList(ColorStateList.valueOf(
                    ContextCompat.getColor(requireContext(), R.color.colorPrimary)));
        }
    }

    private void processGalleryImage(Uri imageUri) {
        if (imageUri == null) {
            Toast.makeText(getContext(), "No image selected", Toast.LENGTH_SHORT).show();
            return;
        }

        if (segmentor == null) {
            Toast.makeText(getContext(), "Model not loaded yet", Toast.LENGTH_SHORT).show();
            return;
        }

        // Show loading indicator
        Activity activity = getActivity();
        if (activity == null) return;

        // Pause camera updates while showing gallery result
        isShowingGalleryResult = true;

        Toast.makeText(getContext(), "Processing image...", Toast.LENGTH_SHORT).show();

        // Use dedicated executor since mBackgroundHandler is null after onPause()
        galleryExecutor.execute(() -> {
            try {
                InputStream inputStream = activity.getContentResolver().openInputStream(imageUri);
                if (inputStream == null) {
                    activity.runOnUiThread(() -> {
                        isShowingGalleryResult = false;
                        Toast.makeText(getContext(), "Failed to open image", Toast.LENGTH_SHORT).show();
                    });
                    return;
                }

                // Decode with options to handle large images
                BitmapFactory.Options options = new BitmapFactory.Options();
                options.inPreferredConfig = Bitmap.Config.ARGB_8888;

                Bitmap bitmap = BitmapFactory.decodeStream(inputStream, null, options);
                inputStream.close();

                if (bitmap == null) {
                    activity.runOnUiThread(() -> {
                        isShowingGalleryResult = false;
                        Toast.makeText(getContext(), "Failed to decode image", Toast.LENGTH_SHORT).show();
                    });
                    return;
                }

                // Scale bitmap to model input size if too large (prevents OOM)
                int maxDim = Math.max(segmentor.getInputWidth(), segmentor.getInputHeight()) * 2;
                if (bitmap.getWidth() > maxDim || bitmap.getHeight() > maxDim) {
                    float scale = (float) maxDim / Math.max(bitmap.getWidth(), bitmap.getHeight());
                    int newWidth = (int) (bitmap.getWidth() * scale);
                    int newHeight = (int) (bitmap.getHeight() * scale);
                    bitmap = Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true);
                }

                final Bitmap inputBitmap = bitmap;
                Bitmap outputImage = segmentor.predict(inputBitmap, 0); // No rotation for gallery images

                // Store for display
                lastProcessedBitmap = outputImage;

                // Auto-save the processed image
                String savedPath = saveProcessedImageSync(outputImage);

                activity.runOnUiThread(() -> {
                    if (mFragmentRender != null) {
                        mFragmentRender.render(
                                outputImage,
                                0, // FPS not applicable for single image
                                segmentor.getLastInferenceTime(),
                                segmentor.getLastPreprocessingTime(),
                                segmentor.getLastPostprocessingTime());

                        if (savedPath != null) {
                            Toast.makeText(getContext(),
                                    String.format("Processed in %.0fms - Saved to %s\nTap screen to resume camera",
                                            (float) segmentor.getLastInferenceTime() / 1_000_000, savedPath),
                                    Toast.LENGTH_LONG).show();
                        } else {
                            Toast.makeText(getContext(),
                                    String.format("Processed in %.0fms\nTap screen to resume camera",
                                            (float) segmentor.getLastInferenceTime() / 1_000_000),
                                    Toast.LENGTH_LONG).show();
                        }
                    }
                });
            } catch (Exception e) {
                e.printStackTrace();
                activity.runOnUiThread(() -> {
                    isShowingGalleryResult = false;
                    Toast.makeText(getContext(), "Error: " + e.getMessage(), Toast.LENGTH_LONG).show();
                });
            }
        });
    }

    private String saveProcessedImageSync(Bitmap bitmap) {
        if (bitmap == null) return null;

        try {
            // Create filename with timestamp
            String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
            String filename = "DepthAnything_" + timestamp + ".jpg";

            // Save to public Pictures folder via MediaStore for gallery visibility
            android.content.ContentValues values = new android.content.ContentValues();
            values.put(android.provider.MediaStore.Images.Media.DISPLAY_NAME, filename);
            values.put(android.provider.MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
            values.put(android.provider.MediaStore.Images.Media.RELATIVE_PATH, "Pictures/DepthAnythingV2");

            Uri uri = requireContext().getContentResolver().insert(
                    android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);

            if (uri != null) {
                OutputStream mediaOut = requireContext().getContentResolver().openOutputStream(uri);
                if (mediaOut != null) {
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 95, mediaOut);
                    mediaOut.close();
                    return "Pictures/DepthAnythingV2/" + filename;
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    private void resumeCamera() {
        isShowingGalleryResult = false;
        lastProcessedBitmap = null;
    }

    private void toggleProfiling() {
        boolean newState = !mFragmentRender.isProfilingEnabled();
        mFragmentRender.setProfilingEnabled(newState);

        if (newState) {
            mFragmentRender.resetProfilingStats();
            btnProfile.setText("Stop");
            btnProfile.setBackgroundTintList(ColorStateList.valueOf(0xFFFF5722)); // Orange when active
            Toast.makeText(getContext(), "Profiling started", Toast.LENGTH_SHORT).show();
        } else {
            btnProfile.setText("Profile");
            btnProfile.setBackgroundTintList(ColorStateList.valueOf(
                    ContextCompat.getColor(requireContext(), R.color.colorPrimary)));
            Toast.makeText(getContext(), "Profiling stopped", Toast.LENGTH_SHORT).show();
        }
    }

    private void shareReport() {
        String report = mFragmentRender.getProfilingReport();

        // Add device info
        report += "\n--- Device Info ---\n";
        report += "Model: " + android.os.Build.MODEL + "\n";
        report += "Manufacturer: " + android.os.Build.MANUFACTURER + "\n";
        report += "Android: " + android.os.Build.VERSION.RELEASE + "\n";

        // Create share intent
        Intent shareIntent = new Intent(Intent.ACTION_SEND);
        shareIntent.setType("text/plain");
        shareIntent.putExtra(Intent.EXTRA_SUBJECT, "Depth Anything V2 - Performance Report");
        shareIntent.putExtra(Intent.EXTRA_TEXT, report);
        startActivity(Intent.createChooser(shareIntent, "Share Report"));
    }

    @Override
    public void onResume() {
        super.onResume();
        startBackgroundThread();

        if (mTextureView.isAvailable()) {
            openCamera();
        } else {
            mTextureView.setSurfaceTextureListener(mSurfaceTextureListener);
        }
    }

    @Override
    public void onPause() {
        closeCamera();
        stopBackgroundThread();
        super.onPause();
    }

    @Override
    public void onDestroy() {
        stopBackgroundThread();
        closeCamera();
        galleryExecutor.shutdown();
        super.onDestroy();
    }

    private void requestCameraPermission() {
        if (shouldShowRequestPermissionRationale(Manifest.permission.CAMERA)) {
            new ConfirmationDialog().show(getChildFragmentManager(), FRAGMENT_DIALOG);
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA);
        }
    }

    private void setUpCameraOutputs() {
        Activity activity = getActivity();
        assert activity != null;
        CameraManager mCameraManager = (CameraManager) activity.getSystemService(Context.CAMERA_SERVICE);
        try {
            for (String cameraId : mCameraManager.getCameraIdList()) {
                CameraCharacteristics characteristics = mCameraManager.getCameraCharacteristics(cameraId);

                Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
                if (facing != null && facing == CameraCharacteristics.LENS_FACING_FRONT) {
                    continue;
                }

                StreamConfigurationMap map = characteristics.get(
                        CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
                if (map == null) {
                    continue;
                }

                Integer sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);
                if (sensorOrientation != null) {
                    mSensorOrientation = sensorOrientation;
                }

                Size[] surfaceSizes = map.getOutputSizes(SurfaceTexture.class);

                assert(segmentor != null);
                int targetWidth, targetHeight;
                if (mSensorOrientation == 90 || mSensorOrientation == 270) {
                    targetWidth = segmentor.getInputHeight();
                    targetHeight = segmentor.getInputWidth();
                } else {
                    targetWidth = segmentor.getInputWidth();
                    targetHeight = segmentor.getInputHeight();
                }

                mPreviewSize = chooseOptimalSize(surfaceSizes, targetWidth, targetHeight);
                mCameraId = cameraId;
                return;
            }
        } catch (CameraAccessException e) {
            e.printStackTrace();
        } catch (NullPointerException e) {
            ErrorDialog.newInstance(getString(R.string.camera_error))
                    .show(getChildFragmentManager(), FRAGMENT_DIALOG);
        }
    }

    private void openCamera() {
        if (ContextCompat.checkSelfPermission(requireActivity(), Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            requestCameraPermission();
            return;
        }

        setUpCameraOutputs();
        Activity activity = requireActivity();
        CameraManager manager = (CameraManager) activity.getSystemService(Context.CAMERA_SERVICE);
        try {
            if (!mCameraOpenCloseLock.tryAcquire(2500, TimeUnit.MILLISECONDS)) {
                throw new RuntimeException("Time out waiting to lock camera opening.");
            }
            manager.openCamera(mCameraId, mStateCallback, mBackgroundHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            throw new RuntimeException("Interrupted while trying to lock camera opening.", e);
        }
    }

    private void closeCamera() {
        try {
            mCameraOpenCloseLock.acquire();
            if (null != mCaptureSession) {
                mCaptureSession.close();
                mCaptureSession = null;
            }
            if (null != mCameraDevice) {
                mCameraDevice.close();
                mCameraDevice = null;
            }
        } catch (InterruptedException e) {
            throw new RuntimeException("Interrupted while trying to lock camera closing.", e);
        } finally {
            mCameraOpenCloseLock.release();
        }
    }

    private void startBackgroundThread() {
        mBackgroundThread = new HandlerThread("CameraBackground");
        mBackgroundThread.start();
        mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
    }

    private void stopBackgroundThread() {
        if (mBackgroundThread != null) {
            mBackgroundThread.quitSafely();
            mBackgroundThread = null;
            mBackgroundHandler = null;
        }
    }

    private void createCameraPreviewSession() {
        try {
            SurfaceTexture texture = mTextureView.getSurfaceTexture();
            assert texture != null;

            texture.setDefaultBufferSize(mPreviewSize.getWidth(), mPreviewSize.getHeight());
            Surface surface = new Surface(texture);

            mPreviewRequestBuilder = mCameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            mPreviewRequestBuilder.addTarget(surface);

            try {
                mCameraDevice.createCaptureSession(List.of(surface), new CameraCapture(), null);
            } catch (CameraAccessException e) {
                e.printStackTrace();
            }
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    static class CompareSizesByArea implements Comparator<Size> {
        @Override
        public int compare(Size lhs, Size rhs) {
            return Long.signum((long) lhs.getWidth() * lhs.getHeight() -
                    (long) rhs.getWidth() * rhs.getHeight());
        }
    }

    public static class ErrorDialog extends DialogFragment {
        private static final String ARG_MESSAGE = "message";

        public static ErrorDialog newInstance(String message) {
            ErrorDialog dialog = new ErrorDialog();
            Bundle args = new Bundle();
            args.putString(ARG_MESSAGE, message);
            dialog.setArguments(args);
            return dialog;
        }

        @NonNull
        @Override
        public Dialog onCreateDialog(Bundle savedInstanceState) {
            final Activity activity = getActivity();
            return new AlertDialog.Builder(activity)
                    .setMessage(requireArguments().getString(ARG_MESSAGE))
                    .setPositiveButton(android.R.string.ok, (dialogInterface, i) -> Objects.requireNonNull(activity).finish())
                    .create();
        }
    }

    public static class ConfirmationDialog extends DialogFragment {
        private ActivityResultLauncher<String> requestPermissionLauncher;

        @Override
        public void onCreate(Bundle savedInstanceState) {
            super.onCreate(savedInstanceState);

            requestPermissionLauncher = registerForActivityResult(
                    new ActivityResultContracts.RequestPermission(),
                    isGranted -> {
                        if (!isGranted) {
                            Activity activity = getActivity();
                            if (activity != null) {
                                activity.finish();
                            }
                        }
                    }
            );
        }

        @NonNull
        @Override
        public Dialog onCreateDialog(Bundle savedInstanceState) {
            final Fragment parent = getParentFragment();
            return new AlertDialog.Builder(getActivity())
                    .setMessage(R.string.request_permission)
                    .setPositiveButton(android.R.string.ok, (dialog, which) -> requestPermissionLauncher.launch(Manifest.permission.CAMERA))
                    .setNegativeButton(android.R.string.cancel,
                            (dialog, which) -> {
                                Activity activity = Objects.requireNonNull(parent).getActivity();
                                if (activity != null) {
                                    activity.finish();
                                }
                            })
                    .create();
        }
    }

    public class CameraCapture extends android.hardware.camera2.CameraCaptureSession.StateCallback {
        @Override
        public void onConfigured(@NonNull android.hardware.camera2.CameraCaptureSession cameraCaptureSession) {
            mCaptureSession = cameraCaptureSession;

            try {
                mPreviewRequestBuilder.set(CaptureRequest.CONTROL_AF_MODE,
                        CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);

                cameraCaptureSession.setRepeatingRequest(mPreviewRequestBuilder.build(), new CameraSession(),
                        mBackgroundHandler);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        @Override
        public void onConfigureFailed(@NonNull android.hardware.camera2.CameraCaptureSession cameraCaptureSession) {
        }
    }

    private class CameraSession extends android.hardware.camera2.CameraCaptureSession.CaptureCallback {

        @Override
        public void onCaptureCompleted(@NonNull CameraCaptureSession session, @NonNull
                CaptureRequest request, @NonNull TotalCaptureResult result) {

            super.onCaptureCompleted(session, request, result);

            if (lastTic == 0) {
                lastTic = System.currentTimeMillis();
            } else {
                long newTic = System.currentTimeMillis();
                if (lastTic != newTic) {
                    fps = 1000.f / (float)(newTic - lastTic);
                }
                lastTic = newTic;
            }

            // Skip frame if inference is already running or showing gallery result
            if (segmentor != null && !isInferenceRunning && !isShowingGalleryResult) {
                isInferenceRunning = true;

                Bitmap mBitmap = mTextureView.getBitmap();
                if (mBitmap == null) {
                    isInferenceRunning = false;
                    return;
                }

                mBackgroundHandler.post(() -> {
                    try {
                        Bitmap outputImage = segmentor.predict(mBitmap, mSensorOrientation);

                        requireActivity().runOnUiThread(() -> {
                            mFragmentRender.render(
                                    outputImage,
                                    fps,
                                    segmentor.getLastInferenceTime(),
                                    segmentor.getLastPreprocessingTime(),
                                    segmentor.getLastPostprocessingTime());
                        });
                    } finally {
                        isInferenceRunning = false;
                    }
                });
            }
        }

        @Override
        public void onCaptureFailed(@NonNull CameraCaptureSession session, @NonNull
                CaptureRequest request, @NonNull CaptureFailure failure) {
            super.onCaptureFailed(session, request, failure);
        }

        @Override
        public void onCaptureProgressed(@NonNull CameraCaptureSession session, @NonNull
                CaptureRequest request, @NonNull CaptureResult partialResult) {
            super.onCaptureProgressed(session, request, partialResult);
        }

        @Override
        public void onCaptureStarted(@NonNull CameraCaptureSession session, @NonNull
                CaptureRequest request, long timestamp, long frameNumber) {
            super.onCaptureStarted(session, request, timestamp, frameNumber);
        }
    }
}
