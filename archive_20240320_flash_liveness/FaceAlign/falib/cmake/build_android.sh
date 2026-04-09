rm -rf android_build
mkdir android_build
cd android_build
cmake -DANDROID_ABI=armeabi-v7a -DANDROID_NDK=/home/yanyu/ndk/android-ndk-r17c -DANDROID_TOOLCHAIN_NAME=arm-linux-androideabi -DANDROID_ARM_NEON=ON -DANDROID_PLATFORM=android-14 -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=/home/yanyu/ndk/android-ndk-r17c/build/cmake/android.toolchain.cmake -DUSE_ANDROID=ON ..
make -j4
