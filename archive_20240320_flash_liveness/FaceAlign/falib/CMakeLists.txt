cmake_minimum_required(VERSION 2.8)

project(fa)
# 定义新变量 USE_ANDROID 判断是否编译android，值为 OFF
# 如果要编译Android，cmake -DUSE_ANDROID=ON ..


# 定义构建类型，取值Debug或Release，Release
IF (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
            "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
ENDIF()

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include)


file(GLOB_RECURSE native_srcs ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp)

if(${USE_ANDROID})
    # 设置ndk的安装目录
    if(NOT DEFINED ENV{NDKROOT})
        # 没有找到NDKROOT环境变量
        message(FATAL_ERROR "not defined environment variable:NDKROOT")
    else()
        message("开始编译android下的so")
        add_library(fa_sttc STATIC nx_fa_export.cpp ${native_srcs} nx_fa_export.h)
        add_library(fa_dymc SHARED nx_fa_export.cpp ${native_srcs} nx_fa_export.h)
    endif()
elseif(${USE_PYTHON})
    message(”开始编译python下的so“)
    add_definitions(-std=c++11)
    add_definitions("-Wall")
    add_compile_options(-fPIC)
    add_compile_options(-c)
    set(BUILD_USE_64BITS on)
    add_library(fa_sttc STATIC libfa_api.cpp ${native_srcs} libfa_api.h)
    add_library(fa_dymc SHARED libfa_api.cpp ${native_srcs} libfa_api.h)
else()
    message(”开始编译linux下的so“)
    add_library(fa_sttc STATIC nx_fa_export.cpp ${native_srcs} nx_fa_export.h)
    add_library(fa_dymc SHARED nx_fa_export.cpp ${native_srcs} nx_fa_export.h)
endif()