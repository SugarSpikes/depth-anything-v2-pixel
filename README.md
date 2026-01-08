# Depth Anything V2 - Android App

<p align="center">
  <img src="https://img.shields.io/badge/Android-3DDC84?style=for-the-badge&logo=android&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow_Lite-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Qualcomm-3253DC?style=for-the-badge&logo=qualcomm&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
</p>

A high-performance Android application for real-time monocular depth estimation using **Depth Anything V2** model. This app leverages Qualcomm's hardware acceleration (QNN GPU/NPU) to achieve ~150ms inference time on Snapdragon devices.

## 🎯 Features

### Core Functionality
- **Real-time Depth Estimation**: Live camera feed with depth map visualization overlay
- **Gallery Processing**: Process images from your gallery and auto-save depth maps
- **Hardware Acceleration**: Optimized for Qualcomm NPU/GPU via QNN delegate
- **Multiple Color Schemes**: 6 depth visualization modes (Turbo, Magma, Viridis, Plasma, Inferno, Grayscale)

### Advanced Features
- **Runtime Profiling**: Toggle to display active hardware backend (QNN GPU/NPU/CPU)
- **Performance Profiling**: Track min/max/avg inference times with detailed statistics
- **Export Reports**: Share performance reports with device info for benchmarking
- **Tap-to-Resume**: Interactive gallery mode with tap-to-dismiss

## 📱 Screenshots & Usage

### Main Interface
- **Camera View**: Real-time depth estimation on live camera feed
- **Stats Overlay**: FPS, inference time, preprocessing/postprocessing breakdown
- **Control Panel**: Colormap selection and action buttons at bottom

### Colormap Selection (Horizontal Scroll)
Choose from 6 depth visualization modes:
- **Turbo** (default): Rainbow gradient from red (far) to blue (near)
- **Magma**: Purple-yellow gradient
- **Viridis**: Blue-green-yellow perceptually uniform
- **Plasma**: Purple-pink-orange gradient
- **Inferno**: Black-red-yellow gradient
- **Grayscale**: Simple black-white depth map

### Action Buttons
1. **Gallery**:
   - Select an image from your phone's gallery
   - Automatically processes and saves depth map to `Pictures/DepthAnythingV2/`
   - Tap screen to resume camera

2. **Runtime**:
   - Toggle to show/hide active hardware backend
   - Displays which accelerator is being used (QNN GPU, QNN NPU, TFLite GPU, or CPU)
   - No performance impact when disabled

3. **Profile**:
   - Start/stop profiling mode
   - Tracks min/max/avg inference times across frames
   - Useful for benchmarking and optimization

4. **Share**:
   - Export performance report with detailed statistics
   - Includes device info (model, manufacturer, Android version)
   - Share via any app (email, messaging, notes, etc.)

## 🚀 Quick Start

### Prerequisites
- Android device with **Android 8.0 (API 26)** or higher
- **Snapdragon processor** (recommended for best performance)
- Android Studio **2023.1.1 or newer** (for building from source)

### Installation

#### Option 1: Pre-built APK
1. Download `SemanticSegmentation-debug.apk` from the releases
2. Enable "Install from Unknown Sources" in Android settings
3. Install the APK

#### Option 2: Build from Source
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/depth-anything-v2-android.git
cd depth-anything-v2-android

# Open in Android Studio
# File → Open → Select project directory

# Build and run
# Click "Run" button or use:
./gradlew assembleDebug

# Install on connected device
adb install -r build/outputs/apk/debug/SemanticSegmentation-debug.apk
```

## 🔧 Technical Details

### Model Specifications
- **Model**: Depth Anything V2 (Float32)
- **Input**: 1 × 518 × 518 × 3 (RGB, normalized [0,1])
- **Output**: 1 × 518 × 518 × 1 (Float32 depth values)
- **Size**: ~95 MB

### Hardware Acceleration
The app supports multiple hardware backends via TensorFlow Lite delegates:

| Backend | Delegate | Typical Inference Time |
|---------|----------|----------------------|
| **QNN GPU** (default) | Qualcomm QNN GPU Backend | ~150ms |
| QNN NPU | Qualcomm QNN HTP Backend | ~200ms |
| TFLite GPU | TensorFlow Lite GPUv2 | ~250ms |
| CPU | XNNPack | ~3000ms+ |

**Default Configuration**: The app uses **QNN GPU** backend by default, matching the optimal configuration found in benchmarking.

### Performance Optimizations
- **Frame Skipping**: Prevents inference queue buildup at high camera frame rates
- **Async Processing**: Inference runs on background thread to maintain UI responsiveness
- **ByteBuffer Rewind Fix**: Critical fix ensuring consistent depth detection across frames
- **Configurable Overlay**: Depth overlay alpha set to 70% for optimal visibility

### Key Files
```
src/main/
├── java/com/quicinc/semanticsegmentation/
│   ├── SemanticSegmentation.java      # Main inference class
│   ├── CameraFragment.java            # Camera handling & UI
│   ├── FragmentRender.java            # Rendering & stats display
│   └── MainActivity.java              # App entry & delegate config
├── java/com/quicinc/tflite/
│   └── TFLiteHelpers.java            # TFLite delegate management
└── assets/
    └── Depth-Anything-V2_float.tflite # Model file
```

## 🎨 Customization

### Changing the Model
To use a different depth estimation model:

1. Replace the model file:
   ```bash
   cp your_model.tflite src/main/assets/Depth-Anything-V2_float.tflite
   ```

2. Update input/output dimensions in `build.gradle`:
   ```gradle
   buildConfigField "int", "IMAGE_WIDTH", "518"
   buildConfigField "int", "IMAGE_HEIGHT", "518"
   ```

3. Update `strings.xml`:
   ```xml
   <string name="tfLiteModelAsset">Depth-Anything-V2_float.tflite</string>
   ```

### Changing Hardware Backend
Modify `MainActivity.java` to select different delegates:

```java
// Option 1: QNN GPU (fastest - ~150ms)
DelegateType[][] qnnGpu = {{DelegateType.QNN_GPU}};

// Option 2: QNN NPU
DelegateType[][] qnnNpu = {{DelegateType.QNN_NPU}};

// Option 3: TFLite GPU
DelegateType[][] gpuOnly = {{DelegateType.GPUv2}};

// Set your preference
DelegateType[][] selectedDelegate = qnnGpu;
```

## 📊 Benchmarking

### Running Performance Tests
1. Open the app
2. Tap **Profile** button to start profiling
3. Let it run for desired duration (e.g., 100 frames)
4. Tap **Profile** again to stop
5. Tap **Share** to export results

### Sample Report Format
```
=== Depth Anything V2 Performance Report ===

Total Frames: 150
Colormap: TURBO
Backend: QNN GPU

--- Inference Time ---
  Average: 148.5 ms
  Min: 132.0 ms
  Max: 167.0 ms
  Std Dev: 8.2 ms

--- Throughput ---
  Average FPS: 6.7
  Max FPS: 7.6
  Min FPS: 6.0

--- Device Info ---
Model: SM-S918B
Manufacturer: samsung
Android: 14
```

## 🛠️ Development

### Project Structure
```
.
├── build.gradle                 # Gradle build configuration
├── gradle.properties           # Gradle properties
├── src/main/
│   ├── AndroidManifest.xml     # App manifest
│   ├── assets/                 # Model files
│   ├── java/                   # Source code
│   ├── res/                    # Resources (layouts, drawables, etc.)
│   │   ├── drawable/          # Vector drawables (app icon)
│   │   ├── layout/            # XML layouts
│   │   ├── mipmap-*/          # App icons
│   │   └── values/            # Strings, colors, themes
│   └── jniLibs/               # Native libraries (QNN, TFLite, OpenCV)
└── README.md
```

### Building for Release
```bash
# Generate signed APK
./gradlew assembleRelease

# Output location
build/outputs/apk/release/SemanticSegmentation-release.apk
```

### Dependencies
- **Android SDK**: 26+ (Android 8.0+)
- **TensorFlow Lite**: 2.x with QNN delegate
- **OpenCV**: 4.x for image processing and colormaps
- **Qualcomm QNN SDK**: For hardware acceleration
- **Camera2 API**: For camera access

## 🐛 Troubleshooting

### Common Issues

**Q: App crashes on launch**
- Check Android version (requires 8.0+)
- Ensure camera permissions are granted
- Verify model file exists in assets folder

**Q: Slow inference (>1000ms)**
- Device may not support QNN delegates
- Fallback to CPU backend is occurring
- Check Runtime profiling to see active backend

**Q: Gallery images not saving**
- Grant storage permissions in Android settings
- Check available storage space
- Look for images in `Pictures/DepthAnythingV2/`

**Q: Depth map not visible**
- Try different colormap modes
- Ensure adequate lighting
- Model works best with varied depth scenes

### Debug Logging
To view detailed logs:
```bash
adb logcat | grep -E "(DepthAnythingV2|TFLite|QNN)"
```

## 📄 License

This project is released under the **BSD-3-Clause License**.

```
Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the conditions in LICENSE are met.
```

## 🙏 Acknowledgments

- **Depth Anything V2**: Original depth estimation model
- **Qualcomm AI Hub**: TensorFlow Lite models and QNN SDK
- **TensorFlow**: TensorFlow Lite framework
- **OpenCV**: Computer vision library for image processing

## 📞 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing issues for solutions
- Refer to Qualcomm AI Hub documentation for model-specific questions

---

**Built with ❤️ for real-time depth estimation on Android**
