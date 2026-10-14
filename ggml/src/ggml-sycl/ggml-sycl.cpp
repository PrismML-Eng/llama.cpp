//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include <filesystem>
#include <mutex>
#include <unordered_map>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <map>
#include <array>
#include <assert.h>
#include <atomic>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <float.h>
#include <limits>
#include <optional>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <regex>
#ifndef _WIN32
#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#endif

#include <sycl/sycl.hpp>
#include <sycl/backend.hpp>
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#include <level_zero/ze_api.h>
#endif
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
#    include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#endif
#if SYCL_EXT_ONEAPI_VIRTUAL_MEM
#    include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#    include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>
#    define GGML_SYCL_SUPPORT_VMM
#endif
#if defined(__INTEL_LLVM_COMPILER)
    #define GGML_SYCL_DMMV_HAS_ESIMD
#endif
#include <sycl/half_type.hpp>

#include "ggml.h"
#include "ggml-sycl.h"
#include "ptq1-t2.hpp"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-sycl/add-id.hpp"
#include "ggml-sycl/backend.hpp"
#include "ggml-sycl/pq2_xmx.hpp"
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/element_wise.hpp"
#include "ggml-sycl/fwht.hpp"
#include "ggml-sycl/w8a8.hpp"
#include "ggml-sycl/gemm.hpp"
#include "ggml-sycl/getrows.hpp"
#include "ggml-sycl/norm.hpp"
#include "ggml-sycl/presets.hpp"
#include "ggml-sycl/quantize.hpp"
#include "ggml-sycl/repeat_back.hpp"
#include "ggml-sycl/set_rows.hpp"
#include "ggml-sycl/set.hpp"
#include "ggml-sycl/dsv4-hc.hpp"
#include "ggml-sycl/lightning-indexer.hpp"
#include "ggml-sycl/conv2d.hpp"
#include "ggml-sycl/conv2d-dw.hpp"
#include "ggml-sycl/conv2d-transpose.hpp"
#include "ggml-sycl/ssm_conv.hpp"
#include "ggml-sycl/sycl_hw.hpp"
#include "ggml-sycl/ssm_scan.hpp"
#include "ggml-sycl/fill.hpp"
#include "ggml-sycl/cumsum.hpp"
#include "ggml-sycl/diag.hpp"
#include "ggml-sycl/opt-step.hpp"
#include "ggml-sycl/solve_tri.hpp"
#include "ggml-sycl/gated_delta_net.hpp"
#include "ggml-sycl/pool.hpp"
#include "ggml-sycl/cross_entropy_loss.hpp"

#define MEM_SIZE_2M	0x00200000
#define MEM_SIZE_1G	0x40000000

static bool g_sycl_loaded = false;
int g_ggml_sycl_debug = 0;
int g_ggml_sycl_enable_optimize = 1;
int g_ggml_sycl_quant_f16_gemm = 1;
int g_ggml_sycl_quant_s4_gemm = 1;
int g_ggml_sycl_quant_w8a8 = 1;
int g_ggml_sycl_enable_graph = 0;
int g_ggml_sycl_enable_dnn = 1;
int g_ggml_sycl_fa_onednn = 1;
int g_ggml_sycl_fa_onednn_max_kv = 0;
int g_ggml_sycl_enable_vmm = 1;
int g_ggml_sycl_enable_fusion = 1;
int g_ggml_sycl_enable_esimd = 1;
int g_ggml_sycl_prioritize_dmmv = 0;
int g_ggml_sycl_use_async_mem_op = 0;
int g_ggml_sycl_use_async_mem_op_requested = 1;
int g_ggml_sycl_use_level_zero_api = 0;
int g_ggml_sycl_enable_flash_attention = 1;
int g_ggml_sycl_dev2dev_memcpy = DEV2DEV_MEMCPY_SYCL;
int g_ggml_sycl_usm_system = 0;
int g_ggml_sycl_enable_host_pinned_mem = 1;

// int8 DPAS execution size of the device's XMX units (8 or 16), as the runtime reports it; 0 without XMX
static int ggml_sycl_dpas_exec_size(const sycl::device & device) {
    namespace matrix = syclex::matrix;
    if (!device.has(sycl::aspect::ext_intel_matrix)) {
        return 0;
    }
    try {
        for (const matrix::combination & c : device.get_info<syclex::info::device::matrix_combinations>()) {
            if (c.atype == matrix::matrix_type::sint8 && c.btype == matrix::matrix_type::sint8) {
                return (int) c.nsize;
            }
        }
    } catch (const sycl::exception &) {
    }
    return 0;
}

static ggml_sycl_device_info ggml_sycl_init() {
    ggml_sycl_device_info info = {};

    // Do not hard crash when there exists no SYCL devices.
    // We want to allow the user to use non-SYCL tools when SYCL is compiled (such as llama-quantize)
    try {
        info.device_count = dpct::dev_mgr::instance().device_count();
    } catch (sycl::exception const &exc) {
        GGML_LOG_INFO("%s: no SYCL device available: %s\n", __func__, exc.what());
        info.device_count = 0;
    }
    if (info.device_count == 0) {
        GGML_LOG_ERROR("%s: failed to initialize: %s\n", GGML_SYCL_NAME, __func__);
        return info;
    }

    GGML_ASSERT(info.device_count <= GGML_SYCL_MAX_DEVICES);

    int64_t total_vram = 0;
/* This is a bit misleading;  reserved for later */
// #if defined(SYCL_USE_XMX)
//     GGML_LOG_INFO("%s: SYCL_USE_XMX: yes\n", __func__);
// #else
//     GGML_LOG_INFO("%s: SYCL_USE_XMX: no\n", __func__);
// #endif
    for (int i = 0; i < info.device_count; ++i) {
        dpct::device_info prop;
        auto & device = dpct::dev_mgr::instance().get_device(i);

        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, device)));

#if !defined(GGML_SYCL_SUPPORT_VMM)
        info.devices[i].vmm = 0;
#else
        info.devices[i].vmm = device.has(sycl::aspect::ext_oneapi_virtual_mem);
        if (info.devices[i].vmm) {
            // NB: SYCL's get_mem_granularity always returns the _minimum_ granularity,
            // but the L0 API requires a larger page size for allocs above 2 MiB and
            // rejects non-multiples with UR_RESULT_ERROR_INVALID_VALUE [sic].
            // Here we clamp it to 2 MiB for simplicity, but other devices may require
            // calling zeVirtualMemQueryPageSize or yet unexposed public API.
            const size_t physical_page = 2ull << 20; // 2 MiB
            info.devices[i].vmm_granularity = std::max<size_t>(
                sycl::ext::oneapi::experimental::get_mem_granularity(
                    device, sycl::context(device)),
                physical_page);
        }
#endif

        info.default_tensor_split[i] = total_vram;
        total_vram += prop.get_global_mem_size();

        info.devices[i].cc =
            100 * prop.get_major_version() + 10 * prop.get_minor_version();
        info.devices[i].nsm = prop.get_max_compute_units() / 16; //16: Number of Xe Cores
        info.devices[i].opt_feature.reorder = device.ext_oneapi_architecture_is(syclex::arch_category::intel_gpu);
        info.devices[i].smpbo = prop.get_local_mem_size();
        info.devices[i].warp_size = WARP_SIZE;
        info.devices[i].usm_system_support = device.has(sycl::aspect::usm_system_allocations);

        info.max_work_group_sizes[i] = prop.get_max_work_group_size();
        info.devices[i].max_wg_per_cu = info.max_work_group_sizes[i] / prop.get_max_compute_units();
        info.devices[i].hw_info = get_device_hw_info(&device);
        info.devices[i].dpas_exec_size = ggml_sycl_dpas_exec_size(device);

        // Only check GPU devices; CPU devices use OpenCL and would otherwise
        // disable Level Zero for the GPUs on systems without ONEAPI_DEVICE_SELECTOR set.
        if (device.is_gpu() && device.default_queue().get_backend() != sycl::backend::ext_oneapi_level_zero) {
            GGML_LOG_WARN("SYCL GPU device %d does not use Level Zero backend, disabling Level Zero memory API\n", i);
            info.ext_oneapi_level_zero = false;
        }

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        if (info.ext_oneapi_level_zero && device.is_gpu() && device.default_queue().get_backend() == sycl::backend::ext_oneapi_level_zero) {
            ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device.default_queue().get_device());
            ze_device_properties_t props = {};
            props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            ze_result_t r = zeDeviceGetProperties(ze_dev, &props);
            if (r == ZE_RESULT_SUCCESS) {
                info.devices[i].l0_device_type_valid = true;
                info.devices[i].l0_discrete_gpu = !(props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED);
            }
        }
#endif
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    // Large buffers can be allocated before ggml_check_sycl() initializes other
    // g_ggml_sycl_enable_* globals, so initialize this one as early as we can.
    g_ggml_sycl_use_level_zero_api =
        info.ext_oneapi_level_zero && ggml_sycl_get_env("GGML_SYCL_USE_LEVEL_ZERO_API", 1);
#else
    g_ggml_sycl_use_level_zero_api = 0;
#endif

    return info;
}

const ggml_sycl_device_info & ggml_sycl_info() {
    static ggml_sycl_device_info info = ggml_sycl_init();
    return info;
}

static void print_device_detail(int id, sycl::device &device, std::string device_type) {

    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(
        dpct::get_device_info(prop, device)));

    std::string version;
    version += std::to_string(prop.get_major_version());
    version += ".";
    version += std::to_string(prop.get_minor_version());

    device_type = std::regex_replace(device_type, std::regex("ext_oneapi_"), "");
    std::string name = std::string(prop.get_name());
    name = std::regex_replace(name, std::regex("\\(R\\)"), "");
    name = std::regex_replace(name, std::regex("\\(TM\\)"), "");

    auto global_mem_size = prop.get_global_mem_size()/1000000;
    GGML_LOG_INFO("|%2d|%19s|%39s|%7s|%7d|%8d|%5d|%6luM|%21s|\n", id, device_type.c_str(),
            name.c_str(), version.c_str(), prop.get_max_compute_units(),
            prop.get_max_work_group_size(), prop.get_max_sub_group_size(),
            global_mem_size, device.get_info<sycl::info::device::driver_version>().c_str());
}

static void print_device_opt_feature(int device_count) {
    GGML_LOG_INFO("SYCL Optimization Feature:\n");
    GGML_LOG_INFO(
        "|ID|        Device Type|Reorder|\n");
    GGML_LOG_INFO(
        "|--|-------------------|-------|\n");
    std::map<std::string, size_t> DeviceNums;
    for (int id = 0; id < device_count; ++id) {
      sycl::device device = dpct::dev_mgr::instance().get_device(id);
      std::string backend_type = get_device_backend_and_type(device);
      int type_id = DeviceNums[backend_type]++;
      std::stringstream device_type;
      device_type << "[" << backend_type << ":" << std::to_string(type_id)
                  << "]";
      std::string device_type_s = device_type.str();
      device_type_s = std::regex_replace(device_type_s, std::regex("ext_oneapi_"), "");
      GGML_LOG_INFO("|%2d|%19s|%7s|\n", id, device_type_s.c_str(),
        ggml_sycl_info().devices[id].opt_feature.reorder ? "Y": "N");
    }

}
void ggml_backend_sycl_print_sycl_devices() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_print_sycl_devices\n");
    int device_count = dpct::dev_mgr::instance().device_count();
    std::map<std::string, size_t> DeviceNums;
    GGML_LOG_INFO("Found %d SYCL devices:\n", device_count);

    GGML_LOG_INFO(
        "|  |                   |                                       |      "
        " |Max    |        |Max  |Global |                     |\n");
    GGML_LOG_INFO(
        "|  |                   |                                       |      "
        " |compute|Max work|sub  |mem    |                     |\n");
    GGML_LOG_INFO(
        "|ID|        Device Type|                                   "
        "Name|Version|units  |group   |group|size   |       Driver version|\n");
    GGML_LOG_INFO(
        "|--|-------------------|---------------------------------------|------"
        "-|-------|--------|-----|-------|---------------------|\n");

    for (int id = 0; id < device_count; ++id) {
      sycl::device device = dpct::dev_mgr::instance().get_device(id);
      std::string backend_type = get_device_backend_and_type(device);
      int type_id = DeviceNums[backend_type]++;
      std::stringstream device_type;
      device_type << "[" << backend_type << ":" << std::to_string(type_id)
                  << "]";
      print_device_detail(id, device, device_type.str());
    }

    print_device_opt_feature(device_count);
}

static const char* dev2dev_int2str(int dev2dev) {
    if (dev2dev == DEV2DEV_MEMCPY_SYCL) {
        return "SYCL API";
    } else if (dev2dev == DEV2DEV_MEMCPY_L0) {
        return "Level Zero API";
    } else if (dev2dev == DEV2DEV_MEMCPY_FORWARD) {
        return "Host Forward";
    } else {
        return "Unknown";
    }
}

// ARC-LAB: use the legacy Level Zero adapter unless the user chose one. On the B580 (oneAPI 2025.3, where V2 is the
// default) plain decode measured +1-1.5% in llama-bench and MTP +0.5%, outputs identical. Set at library load, before
// any SYCL platform query; SYCL_UR_USE_LEVEL_ZERO_V2=1 in the environment keeps V2.
namespace {
struct ggml_sycl_l0_adapter_default {
    ggml_sycl_l0_adapter_default() {
        if (getenv("SYCL_UR_USE_LEVEL_ZERO_V2") == nullptr) {
#ifdef _WIN32
            _putenv_s("SYCL_UR_USE_LEVEL_ZERO_V2", "0");
#else
            setenv("SYCL_UR_USE_LEVEL_ZERO_V2", "0", 0);
#endif
        }
    }
} ggml_sycl_l0_adapter_default_init;
}

static void ggml_check_sycl() try {
    static bool initialized = false;

    if (!initialized) {
        g_ggml_sycl_debug = ggml_sycl_get_env("GGML_SYCL_DEBUG", 0);
        g_ggml_sycl_enable_optimize = ggml_sycl_get_env("GGML_SYCL_ENABLE_OPT", 1);
        g_ggml_sycl_quant_f16_gemm = ggml_sycl_get_env("GGML_SYCL_QUANT_F16_GEMM", 1);
        g_ggml_sycl_quant_s4_gemm = ggml_sycl_get_env("GGML_SYCL_QUANT_S4_GEMM", 1);
        g_ggml_sycl_quant_w8a8 = ggml_sycl_get_env("GGML_SYCL_QUANT_W8A8", 1);
        g_ggml_sycl_enable_graph = ggml_sycl_get_env("GGML_SYCL_ENABLE_GRAPH", 0);
        g_ggml_sycl_enable_dnn = ggml_sycl_get_env("GGML_SYCL_ENABLE_DNN", 1);
        g_ggml_sycl_fa_onednn = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN", 1);
        g_ggml_sycl_fa_onednn_max_kv = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN_MAX_KV", 0);
        g_ggml_sycl_enable_vmm = ggml_sycl_get_env("GGML_SYCL_ENABLE_VMM", 1);
        g_ggml_sycl_enable_fusion = ggml_sycl_get_env("GGML_SYCL_ENABLE_FUSION", 1);
        g_ggml_sycl_enable_esimd = ggml_sycl_get_env("GGML_SYCL_ENABLE_ESIMD", 1);
        g_ggml_sycl_prioritize_dmmv = ggml_sycl_get_env("GGML_SYCL_PRIORITIZE_DMMV", 0);

        g_ggml_sycl_dev2dev_memcpy = ggml_sycl_get_env("GGML_SYCL_DEV2DEV_MEMCPY", DEV2DEV_MEMCPY_SYCL);
        if (g_ggml_sycl_use_level_zero_api == 0) {
            g_ggml_sycl_dev2dev_memcpy = DEV2DEV_MEMCPY_SYCL;
        }

#ifdef SYCL_FLASH_ATTN
        g_ggml_sycl_enable_flash_attention = ggml_sycl_get_env("GGML_SYCL_ENABLE_FLASH_ATTN", 1);
#else
        g_ggml_sycl_enable_flash_attention = 0;
#endif

        g_ggml_sycl_usm_system = ggml_sycl_get_env("GGML_SYCL_USM_SYSTEM", 0);
        g_ggml_sycl_enable_host_pinned_mem =
            ggml_sycl_get_env("GGML_SYCL_ENABLE_HOST_PINNED_MEM", 1);

        GGML_SYCL_DEBUG("[SYCL] call ggml_check_sycl\n");

        GGML_LOG_INFO("Build with Macros:\n");
#if defined(GGML_SYCL_DNNL)
        GGML_LOG_INFO("  GGML_SYCL_DNNL: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_DNNL: no\n");
#endif

#if defined(GGML_SYCL_F16)
        GGML_LOG_INFO("  GGML_SYCL_F16: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_F16: no\n");
#endif

#if defined(GGML_SYCL_FORCE_MMQ)
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: no\n");
#endif

#if defined(GGML_SYCL_GRAPH)
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: no\n");
#endif

#if defined(GGML_SYCL_SUPPORT_LEVEL_ZERO_API)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO_API: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO_API: no\n");
#endif
#if defined(GGML_SYCL_SUPPORT_VMM)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_VMM: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_VMM: no\n");
#endif

        GGML_LOG_INFO("Running with Environment Variables:\n");
        GGML_LOG_INFO("  GGML_SYCL_DEBUG: %d\n", g_ggml_sycl_debug);

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_LOG_INFO("  GGML_SYCL_DEV2DEV_MEMCPY: %d (%s)\n", g_ggml_sycl_dev2dev_memcpy, dev2dev_int2str(g_ggml_sycl_dev2dev_memcpy));
#else
        GGML_LOG_INFO("  GGML_SYCL_DEV2DEV_MEMCPY: %d (%s), enable to SYCL API since missing GGML_SYCL_SUPPORT_LEVEL_ZERO_API\n",
                      g_ggml_sycl_dev2dev_memcpy, dev2dev_int2str(g_ggml_sycl_dev2dev_memcpy));
#endif

#if defined(GGML_SYCL_DNNL)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_DNN: %d\n", g_ggml_sycl_enable_dnn);
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN: %d\n", g_ggml_sycl_fa_onednn);
#else
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_DNN: DNN disabled by compile flag\n");
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN: %d\n", g_ggml_sycl_fa_onednn);
#endif
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN_MAX_KV: %d\n", g_ggml_sycl_fa_onednn_max_kv);
#ifdef SYCL_FLASH_ATTN
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FLASH_ATTN: %d\n", g_ggml_sycl_enable_flash_attention);
#else
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FLASH_ATTN: %d disabled by compile flag\n",
            g_ggml_sycl_enable_flash_attention);
#endif

#ifdef GGML_SYCL_GRAPH
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_GRAPH: %d\n", g_ggml_sycl_enable_graph);
#else
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_GRAPH: graph disabled by compile flag\n");
#endif

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_OPT: %d\n", g_ggml_sycl_enable_optimize);
        GGML_LOG_INFO("  GGML_SYCL_QUANT_F16_GEMM: %d\n", g_ggml_sycl_quant_f16_gemm);
        GGML_LOG_INFO("  GGML_SYCL_QUANT_S4_GEMM: %d\n", g_ggml_sycl_quant_s4_gemm);

#if defined(GGML_SYCL_SUPPORT_VMM)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: %d\n", g_ggml_sycl_enable_vmm);
#else
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: virtual memory extension is not available\n");
#endif

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FUSION: %d\n", g_ggml_sycl_enable_fusion);

#if defined(__INTEL_LLVM_COMPILER)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_ESIMD: %d\n", g_ggml_sycl_enable_esimd);
#else
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_ESIMD: %d disabled by compile flag\n", g_ggml_sycl_enable_esimd);
#endif

        GGML_LOG_INFO("  GGML_SYCL_PRIORITIZE_DMMV: %d\n", g_ggml_sycl_prioritize_dmmv);

        g_ggml_sycl_use_async_mem_op_requested = ggml_sycl_get_env("GGML_SYCL_USE_ASYNC_MEM_OP", 1);
        GGML_LOG_INFO("  GGML_SYCL_USE_ASYNC_MEM_OP: %d\n", g_ggml_sycl_use_async_mem_op_requested);

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_LOG_INFO("  GGML_SYCL_USE_LEVEL_ZERO_API: %d\n", g_ggml_sycl_use_level_zero_api);
#else
        GGML_LOG_INFO("  GGML_SYCL_USE_LEVEL_ZERO_API: Disable Level Zero API usage by compile flag\n");
#endif

        GGML_LOG_INFO("  GGML_SYCL_USM_SYSTEM: %d\n", g_ggml_sycl_usm_system);
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_HOST_PINNED_MEM: %d\n", g_ggml_sycl_enable_host_pinned_mem);

/* NOT REMOVE, keep it for next optimize for XMX.
#if defined(SYCL_USE_XMX)
        fprintf(stderr, "%s: SYCL_USE_XMX: yes\n", __func__);
#else
        fprintf(stderr, "%s: SYCL_USE_XMX: no\n", __func__);
#endif
*/
        // Async USM allocation/free is also useful outside the graph path: it avoids the host waits in the reorder
        // staging path while preserving queue ordering semantics. Graph support still depends on the extension being
        // available, but it no longer needs to control the non-graph fast path.
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
        g_ggml_sycl_use_async_mem_op = g_ggml_sycl_use_async_mem_op_requested || g_ggml_sycl_enable_graph;
        if (g_ggml_sycl_use_async_mem_op) {
            for (unsigned int i = 0; i < dpct::dev_mgr::instance().device_count(); ++i) {
                if (!dpct::dev_mgr::instance().get_device(i).has(sycl::aspect::ext_oneapi_async_memory_alloc)) {
                    g_ggml_sycl_use_async_mem_op = 0;
                    break;
                }
            }
        }
#endif
        if (CHECK_TRY_ERROR(g_all_sycl_device_count =
                            dpct::dev_mgr::instance().device_count()) != 0) {
            initialized = true;
            g_sycl_loaded = false;
            return;
        }
        GGML_ASSERT(g_all_sycl_device_count <= GGML_SYCL_MAX_DEVICES);

        initialized = true;
        g_sycl_loaded = true;
        ggml_backend_sycl_print_sycl_devices();
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/*
device_index: device index from 0 to n (continue numbers).
    It is used for device select/set in SYCL backend internal data structure.
*/
inline void check_allow_gpu_index(const int device_index) {
  if (device_index >= ggml_sycl_info().device_count) {
    char error_buf[256];
    snprintf(
        error_buf,
        sizeof(error_buf),
        "%s error: device_index:%d is out of range: [0-%d]",
        __func__,
        device_index,
        ggml_sycl_info().device_count - 1);
    GGML_LOG_ERROR("%s\n", error_buf);
    assert(false);
  }
}

GGML_API void ggml_backend_sycl_get_gpu_list(int *id_list, int max_len) try {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_gpu_list\n");
    for(int i=0;i<max_len;i++) id_list[i] = -1;

    for (int i=0;i< ggml_sycl_info().device_count;i++){
        if (i>=max_len) break;
        id_list[i] = i;
    }
    return;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

inline void free_aligned_mem_host(void * memblock) {
#ifdef _WIN32
    _aligned_free(memblock);
#else
    free(memblock);
#endif
}

// sycl buffer

struct ggml_backend_sycl_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    queue_ptr stream;
    std::string name;
    optimize_feature opt_feature;
    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    bool is_usm_system;

    ggml_backend_sycl_buffer_context(int device, void * dev_ptr, queue_ptr stream, bool is_usm_system) :
        device(device), dev_ptr(dev_ptr), stream(stream), is_usm_system(is_usm_system) {
            check_allow_gpu_index(device);
            name = (GGML_SYCL_NAME + std::to_string(device));
            opt_feature = ggml_sycl_info().devices[device].opt_feature;
        }

    ~ggml_backend_sycl_buffer_context() {
        if (dev_ptr != nullptr) {
            ggml_sycl_set_device(device);
            if (is_usm_system)
                free_aligned_mem_host(dev_ptr);
            else
                SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(dev_ptr, *stream)));
        }

        //release extra used by tensors
        for (ggml_tensor_extra_gpu * extra : tensor_extras) {
            release_extra_gpu(extra);
        }

    }
};

static const char * ggml_backend_sycl_buffer_type_get_name(ggml_backend_buffer_type_t buft);

static bool ggml_backend_buffer_is_sycl(ggml_backend_buffer_t buffer) {
    return buffer->buft->iface.get_name == ggml_backend_sycl_buffer_type_get_name;
}

static void
ggml_backend_sycl_buffer_free_buffer(ggml_backend_buffer_t buffer) try {
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    ggml_sycl_set_device(ctx->device);

    delete ctx;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void * ggml_backend_sycl_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

static enum ggml_status
ggml_backend_sycl_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                     ggml_tensor *tensor) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor, "\n").c_str());
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *)buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    if (g_ggml_sycl_enable_optimize) {
        // set reorder extra buffer based on supported type
        switch (tensor->type) {
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
            case GGML_TYPE_PQ2_0:
            case GGML_TYPE_PTQ1_0:{
                ggml_tensor_extra_gpu * extra = new ggml_tensor_extra_gpu{};
                tensor->extra                 = extra;
                ctx->tensor_extras.push_back(extra);
                break;
            }
            default:
                break;
        }
    }

    if (ggml_is_quantized(tensor->type)) {
        // initialize padding to 0 to avoid possible NaN values
        size_t original_size = ggml_nbytes(tensor);
        size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size && tensor->view_src == nullptr) {
            SYCL_CHECK(CHECK_TRY_ERROR(ctx->stream->memset(
                (char *)tensor->data + original_size, 0,
                padded_size - original_size).wait()));
        }
    }
    return GGML_STATUS_SUCCESS;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// ARC-LAB: GGML_SYCL_SYNC_TRACE=1: every ggml_backend_sycl_synchronize and blocking buffer_set_tensor is counted under its
// cause: the scheduler's tag (ggml_sync_trace_set_cause: split input <name>, backend switch), else the first named caller
// outside libggml (e.g. llama_get_logits_ith). Printed per 64 graphs of >= 2000 nodes (target graphs: ~ per MTP cycle).
struct ggml_sycl_sync_trace {
    const bool on = getenv("GGML_SYCL_SYNC_TRACE") && atoi(getenv("GGML_SYCL_SYNC_TRACE")) != 0;
    std::mutex mtx;
    int        n_big = 0, n_small = 0;
    std::map<std::string, long>                         syncs;
    std::map<std::string, std::pair<long, size_t>>     uploads;
    std::unordered_map<void *, std::string>             names;  // return address -> caller label

    std::string caller() {
        const char * c = ggml_sync_trace_get_cause();
        if (c) {
            return c;
        }
#ifdef _WIN32
        return std::string("?"); // no caller lookup on Windows
#else
        void * fr[24];
        const int n = backtrace(fr, 24);
        std::string label, last;
        int named = 0;
        for (int i = 1; i < n && named < 2; ++i) {
            auto it = names.find(fr[i]);
            std::string nm;
            if (it != names.end()) {
                nm = it->second;
            } else {
                Dl_info di{};
                if (dladdr(fr[i], &di) && di.dli_sname && di.dli_fname && !strstr(di.dli_fname, "libggml")) {
                    int st = 0;
                    char * dm = abi::__cxa_demangle(di.dli_sname, nullptr, nullptr, &st);
                    nm = dm && st == 0 ? std::string(dm) : std::string(di.dli_sname);
                    free(dm);
                    const size_t paren = nm.find('(');
                    if (paren != std::string::npos) {
                        nm.resize(paren);
                    }
                }
                names[fr[i]] = nm;
            }
            if (!nm.empty() && nm != last) {
                label += (label.empty() ? "" : " <- ") + nm;
                last = nm;
                named++;
            }
        }
        return label.empty() ? std::string("?") : label;
#endif
    }
    void note_sync() {
        std::lock_guard<std::mutex> lock(mtx);
        syncs[caller()]++;
    }
    void note_upload(size_t bytes) {
        std::lock_guard<std::mutex> lock(mtx);
        auto & u = uploads[caller()];
        u.first++;
        u.second += bytes;
    }
    void note_graph(const ggml_cgraph * g) {
        std::lock_guard<std::mutex> lock(mtx);
        if (g->n_nodes < 2000) {
            n_small++;
            return;
        }
        if (++n_big < 64) {
            return;
        }
        long ns = 0, nu = 0;
        size_t nb = 0;
        for (auto & kv : syncs) ns += kv.second;
        for (auto & kv : uploads) { nu += kv.second.first; nb += kv.second.second; }
        fprintf(stderr, "sync-trace: %d big graphs (+%d smaller), per big graph: %.1f synchronizes, %.1f blocking uploads (%.0f bytes)\n",
                n_big, n_small, (double) ns / n_big, (double) nu / n_big, (double) nb / n_big);
        std::vector<std::pair<long, std::string>> v;
        for (auto & kv : syncs) v.push_back({ kv.second, kv.first });
        std::sort(v.rbegin(), v.rend());
        for (auto & e : v) fprintf(stderr, "sync-trace:   sync   %6.2f  %s\n", (double) e.first / n_big, e.second.c_str());
        std::vector<std::pair<long, std::string>> w;
        for (auto & kv : uploads) w.push_back({ kv.second.first, kv.first });
        std::sort(w.rbegin(), w.rend());
        for (auto & e : w) fprintf(stderr, "sync-trace:   upload %6.2f  %8.0f B  %s\n", (double) e.first / n_big,
                                   (double) uploads[e.second].second / n_big, e.second.c_str());
        n_big = n_small = 0;
        syncs.clear();
        uploads.clear();
    }
};
static ggml_sycl_sync_trace g_sync_trace;

// ARC-LAB: per-token accounting of synchronous buffer_set_tensor calls, printed with the GGML_SYCL_HOST_TIMING report
static double g_st_ms = 0; static long g_st_calls = 0; static size_t g_st_bytes = 0;
static void ggml_sycl_note_set_tensor(size_t bytes, double ms) { g_st_ms += ms; g_st_calls++; g_st_bytes += bytes; }

static void ggml_backend_sycl_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                ggml_tensor *tensor,
                                                const void *data, size_t offset,
                                                size_t size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    ggml_sycl_set_device(ctx->device);
    auto stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_queue());
    // ARC-LAB: GGML_SYCL_HOST_TIMING counts these (llama.cpp writes every per-token input through here, synchronously)
    struct st_timer {
        size_t bytes; std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        ~st_timer() { ggml_sycl_note_set_tensor(bytes, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count()); }
    } st_guard{ size };
    if (g_sync_trace.on) {
        g_sync_trace.note_upload(size);
    }
    // ARC-LAB: GGML_SYCL_FAST_SET_TENSOR=1: small writes (<= 4 MB, i.e. per-token inputs, not weights) skip the device-wide
    // wait and the malloc'd bounce buffer (the mmap workaround is for weights loaded from disk): one pinned staging copy and
    // one wait on this queue. Still synchronous for the caller.
    const int fast_st = ggml_sycl_knob("GGML_SYCL_FAST_SET_TENSOR", 0);  // 2 = keep the device-wide wait
    if (size <= (4u << 20) && fast_st) {
        if (fast_st == 2) {
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::dev_mgr::instance().get_device(ctx->device).queues_wait_and_throw()));
        }
        static char * pinned = nullptr;
        if (!pinned) {
            pinned = (char *) sycl::malloc_host(4u << 20, *stream);
        }
        if (pinned) {
            memcpy(pinned, data, size);
            SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy((char *) tensor->data + offset, pinned, size).wait()));
            return;
        }
    }
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::dev_mgr::instance().get_device(ctx->device).queues_wait_and_throw()));
#ifndef _WIN32
    // Note: Use host buffer to save the data from mmap(), then copy to device. It's workaround for mmap() issue on PVC GPU.
    // This function will be called during load model from disk. Use memory buffer replace dynamic won't save more time and brings potential memory leak risk here.
    char * host_buf = (char *) malloc(size);
    memcpy(host_buf, data, size);
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy((char *) tensor->data + offset, host_buf, size).wait()));
    free(host_buf);
#else
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy((char *) tensor->data + offset, data, size).wait()));
#endif
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                const ggml_tensor *tensor,
                                                void *data, size_t offset,
                                                size_t size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;

    ggml_sycl_set_device(ctx->device);
    auto stream = dpct::dev_mgr::instance().get_device(ctx->device).default_queue();

    SYCL_CHECK(CHECK_TRY_ERROR(
        stream.memcpy(data, (const char *)tensor->data + offset, size)
            .wait()));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
static bool ggml_sycl_is_l0_discrete_gpu(int device) {
    return ggml_sycl_info().devices[device].l0_discrete_gpu;
}
#endif

static void dev2dev_memcpy(int device_dst, sycl::queue &q_dst, int device_src, sycl::queue &q_src, void *ptr_dst,
                    const void *ptr_src, size_t size) {

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    if (g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_L0) {
        // Use Level Zero direct copy for dGPU-to-dGPU transfers.
        const bool l0_copy_supported =
            ggml_sycl_is_l0_discrete_gpu(device_dst) && ggml_sycl_is_l0_discrete_gpu(device_src);
        if (g_ggml_sycl_use_level_zero_api && l0_copy_supported) {
            auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_context());
            auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_device());
            ze_command_queue_desc_t cq_desc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0,
                                            0, ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
            ze_command_list_handle_t cl;
            ze_result_t r = zeCommandListCreateImmediate(ze_ctx, ze_dev, &cq_desc, &cl);
            if (r == ZE_RESULT_SUCCESS) {
                GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by L0\n");
                r = zeCommandListAppendMemoryCopy(cl, ptr_dst, ptr_src, size, nullptr, 0, nullptr);
                zeCommandListDestroy(cl);
                if (r == ZE_RESULT_SUCCESS) {
                    return;
                }
            }
        }
    }
#endif

    if (g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_SYCL) {
        if (q_dst.get_device().ext_oneapi_can_access_peer(q_src.get_device(),
                                                          sycl::ext::oneapi::peer_access::access_supported)) {
            GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by SYCL\n");
            SYCL_CHECK(CHECK_TRY_ERROR(q_dst.memcpy(ptr_dst, ptr_src, size).wait()));
            return;
        }
    }

    // Host-staged copy
    if(g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_FORWARD) {
        GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by host forward for setting GGML_SYCL_DEV2DEV_MEMCPY=2\n");
    } else {
        GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by host forward for SYCL/L0 fallback\n");
    }
    char *host_buf = (char *)malloc(size);
    q_src.memcpy(host_buf, (const char *)ptr_src, size).wait();
    q_dst.memcpy((char *)ptr_dst, host_buf, size).wait();
    free(host_buf);
}

static bool
ggml_backend_sycl_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                    const ggml_tensor *src,
                                    ggml_tensor *dst) try {
    bool is_cpy_supported = ggml_backend_buffer_is_sycl(src->buffer);
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": dst", dst).c_str());
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" src", src).c_str());
    GGML_SYCL_DEBUG(" is_cpy_supported=%d\n", is_cpy_supported);
    if (is_cpy_supported) {
        ggml_backend_sycl_buffer_context * src_ctx = (ggml_backend_sycl_buffer_context *)src->buffer->context;
        ggml_backend_sycl_buffer_context * dst_ctx = (ggml_backend_sycl_buffer_context *)dst->buffer->context;

        ggml_sycl_set_device(src_ctx->device);
        /*
        DPCT1009:198: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(src_ctx->device).queues_wait_and_throw()));
        ggml_sycl_set_device(dst_ctx->device);
        /*
        DPCT1009:199: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(dst_ctx->device).queues_wait_and_throw()));
        /*
        DPCT1009:200: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */

        queue_ptr stream_dst = dst_ctx->stream;
        queue_ptr stream_src = src_ctx->stream;
        size_t size = ggml_nbytes(src);

        //todo. it's dirty solutino to walkaroud known issue:device2device cross GPUs.
        dev2dev_memcpy(dst_ctx->device, *stream_dst, src_ctx->device, *stream_src, dst->data, src->data, size);

//todo, it's known issue：error in device2device cross GPUs. reused when the issue is fixed. DON"T remove
#if 0
        SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy(
            (char *)dst->data, (const char *)src->data, size).wait()));

        /*
        DPCT1009:201: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(dst_ctx->device).queues_wait_and_throw()));
#endif
        return true;
    }
    return false;
    GGML_UNUSED(buffer);
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static void ggml_backend_sycl_buffer_clear(ggml_backend_buffer_t buffer,
                                           uint8_t value) try {
    GGML_SYCL_DEBUG("[SYCL] call %s: size=%zu\n", __func__, buffer->size);
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;

    ggml_sycl_set_device(ctx->device);
    queue_ptr stream = ctx->stream;
    SYCL_CHECK(
        CHECK_TRY_ERROR(dpct::get_current_device().queues_wait_and_throw()));

    constexpr size_t MAX_CHUNK = 2ULL << 30;  // 2 GiB
    for (size_t off = 0; off < buffer->size; off += MAX_CHUNK) {
        size_t chunk = std::min(buffer->size - off, MAX_CHUNK);
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .memset(static_cast<char*>(ctx->dev_ptr) + off, value, chunk)
                .wait()
        ));
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value,
                                                   size_t offset, size_t size) {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu value=%u\n", size, offset, value);
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;
    SYCL_CHECK(ggml_sycl_set_device(ctx->device));
    auto stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_queue());
    if (size == 0) {
        return;  // Nothing to do
    }
    if (tensor->data == nullptr) {
        GGML_ABORT("Error: Tensor data pointer is null.\n");
    }
    void * target_ptr = static_cast<char *>(tensor->data) + offset;
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).memset(target_ptr, value, size)));
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).wait()));
}

static void ggml_backend_sycl_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    if (buffer == nullptr) {
        return;
    }

    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;

    if (ctx != nullptr) {
        for (ggml_tensor_extra_gpu * extra : ctx->tensor_extras) {
            release_extra_gpu(extra);
        }
        ctx->tensor_extras.clear();  // reset the tensor_extras vector
    }
}

static const ggml_backend_buffer_i ggml_backend_sycl_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_sycl_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_sycl_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_sycl_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_sycl_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_sycl_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_sycl_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_sycl_buffer_clear,
    /* .reset           = */ ggml_backend_sycl_buffer_reset,
};

// sycl buffer type
struct ggml_backend_sycl_buffer_type_context {
    int device;
    std::string name;

    // each buffer type has its own stream
    queue_ptr stream = nullptr;
};

static const char * ggml_backend_sycl_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_sycl_buffer_type_context * ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool check_usm_system(int device, size_t size) {
    bool use_usm_system = g_ggml_sycl_usm_system && size >= ((size_t)4 * MEM_SIZE_1G);

    if (use_usm_system && !ggml_sycl_info().devices[device].usm_system_support) {
        GGML_LOG_INFO("Device does not support USM system allocations\n");
        use_usm_system = false;
    }

    return use_usm_system;
}

inline void * aligned_malloc_host(size_t alignment, size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    return aligned_alloc(alignment, size);
#endif
}

static ggml_backend_buffer_t
ggml_backend_sycl_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                           size_t size) try {
    ggml_check_sycl();

    ggml_backend_sycl_buffer_type_context * buft_ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;
    ggml_sycl_set_device(buft_ctx->device);
    const queue_ptr stream = buft_ctx->stream;
    size = std::max(size, (size_t)1); // syclMalloc returns null for size 0
    /*
    Alignment below ensures best performance. While in theory it could lead to
    wasting memory, this is acceptable because in practice only few buffers are
    allocated and even less exceed the minimum size accepted here for USM system
    allocations.
     */
    size_t alignment = MEM_SIZE_2M;
    size_t aligned_size = ((size + alignment - 1) / alignment) * alignment;
    bool use_usm_system = check_usm_system(buft_ctx->device, aligned_size);

    void * dev_ptr;
    if (use_usm_system) {
        GGML_SYCL_DEBUG("[SYCL] allocating %zu Bytes with USM system\n", size);
        dev_ptr = (void *)aligned_malloc_host(alignment, aligned_size);
        if (!dev_ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
            return nullptr;
        }
    } else {
        SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (void *)ggml_sycl_malloc_device(size, *stream)));
        if (!dev_ptr) {
          GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on device\n", __func__, size);
          return nullptr;
        }
    }
    ggml_backend_sycl_buffer_context * ctx = new  ggml_backend_sycl_buffer_context(buft_ctx->device, dev_ptr, buft_ctx->stream, use_usm_system);
    return ggml_backend_buffer_init(buft, ggml_backend_sycl_buffer_interface, ctx, size);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static size_t ggml_backend_sycl_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return SYCL_BUFFER_ALIGNMENT;
    GGML_UNUSED(buft);
}

static size_t ggml_backend_sycl_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    return dpct::get_current_device().get_max_mem_alloc_size();

    GGML_UNUSED(buft);
}

static bool ggml_sycl_device_has_dpas16(int device);

// GGML_SYCL_DISABLE_XMX=1 keeps PQ2_0/PTQ1_0 off the XMX path (and its layout), =0 keeps them on it. Unset, Xe3
// leaves them to the MMVQ reorder: on an Arc B390 (PTQ1_0 weights) that is +16% decode and +6% prompt with ~0.9 GB less
// memory, at a higher but still small KLD (6.7e-4 vs 2.4e-4)
static bool ggml_sycl_xmx_disabled(int device) {
    static const char * env = getenv("GGML_SYCL_DISABLE_XMX");
    return env ? atoi(env) != 0 : ggml_sycl_device_is_xe3(device);
}

// PTQ1_0 weights are expanded into the 34-byte PQ2_0 XMX blocks on first use (pq2_xmx.hpp), so on devices
// that run that path their allocation reserves room for the expanded form
static bool ggml_sycl_ptq1_xmx_expands(int device, const ggml_tensor * tensor) {
    return tensor->type == GGML_TYPE_PTQ1_0 && g_ggml_sycl_enable_optimize && !ggml_sycl_xmx_disabled(device) &&
           tensor->ne[2] == 1 && tensor->ne[3] == 1 && ggml_sycl_pq2_xmx_supports_ne0(tensor->ne[0]) &&
           ggml_sycl_device_has_dpas16(device);
}

static size_t ggml_sycl_ptq1_xmx_bytes(const ggml_tensor * tensor) {
    return (size_t) (ggml_nelements(tensor) / QK_PTQ1_0) * sizeof(block_pq2_0);
}

static size_t ggml_backend_sycl_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    size_t size = ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }
    // ARC-LAB: room for the in-place TernSYCL 2-bit repack (GGML_SYCL_PTQ1_T2)
    if ((tensor->type == GGML_TYPE_PTQ1_0 || tensor->type == GGML_TYPE_PQ2_0) && tensor->ne[2] == 1 && tensor->ne[3] == 1 &&
        ggml_sycl_t2_wants(tensor->name, tensor->ne[0], tensor->ne[1])) {
        size = std::max(size, ggml_sycl_t2_bytes(tensor->ne[0], tensor->ne[1]));
    }

    const auto * buft_ctx = (const ggml_backend_sycl_buffer_type_context *) buft->context;
    if (ggml_sycl_ptq1_xmx_expands(buft_ctx->device, tensor)) {
        size = std::max(size, ggml_sycl_ptq1_xmx_bytes(tensor));
    }

    return size;
}

static const ggml_backend_buffer_type_i ggml_backend_sycl_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_sycl_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_sycl_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_sycl_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_sycl_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_sycl_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);


    auto dev_count = ggml_backend_sycl_get_device_count();

    if (device>=dev_count or device<0) {
        GGML_LOG_ERROR("ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call ggml_backend_sycl_set_single_device()\n",
            device, dev_count-1);
        GGML_ASSERT(device<dev_count);
    }
    static struct ggml_backend_buffer_type ggml_backend_sycl_buffer_types[GGML_SYCL_MAX_DEVICES];

    static bool ggml_backend_sycl_buffer_type_initialized = false;

    if (!ggml_backend_sycl_buffer_type_initialized) {
        for (int i = 0; i < dev_count; i++) {
            auto & device_i = dpct::dev_mgr::instance().get_device(i);
            queue_ptr stream = &(device_i.default_queue());
            ggml_backend_sycl_buffer_types[i] = {
                /* .iface    = */ ggml_backend_sycl_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), i),
                /* .context  = */ new ggml_backend_sycl_buffer_type_context{i, GGML_SYCL_NAME + std::to_string(i), stream},
            };
        }
        ggml_backend_sycl_buffer_type_initialized = true;
    }
    return &ggml_backend_sycl_buffer_types[device];
}

static ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(ggml_backend_sycl_context * ctx) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_buffer_type\n");

    int device = ctx->device;
    if (device>=ggml_sycl_info().device_count or device<0) {
        GGML_LOG_ERROR("ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call ggml_backend_sycl_set_single_device()\n",
            device, ggml_sycl_info().device_count-1);
        GGML_ASSERT(device<ggml_sycl_info().device_count);
    }
    static struct ggml_backend_buffer_type ggml_backend_sycl_buffer_types[GGML_SYCL_MAX_DEVICES];

    static bool ggml_backend_sycl_buffer_type_initialized = false;

    if (!ggml_backend_sycl_buffer_type_initialized) {
        for (int i = 0; i < ggml_sycl_info().device_count; i++) {
            ggml_backend_sycl_buffer_types[i] = {
                /* .iface    = */ ggml_backend_sycl_buffer_type_interface,
                /* .device   = */ nullptr,
                /* .context  = */ new ggml_backend_sycl_buffer_type_context{i, GGML_SYCL_NAME + std::to_string(i), ctx->stream(i, 0)},
            };
        }
        ggml_backend_sycl_buffer_type_initialized = true;
    }
    return &ggml_backend_sycl_buffer_types[device];
}

// sycl split buffer

static int64_t get_row_rounding(ggml_type type, const std::array<float, GGML_SYCL_MAX_DEVICES> & tensor_split) {
    int64_t min_compute_capability = INT_MAX;
    int64_t max_compute_capability = INT_MIN;
    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        if (tensor_split[i] < (i + 1 < ggml_sycl_info().device_count ? tensor_split[i + 1] : 1.0f)) {
            if (min_compute_capability > ggml_sycl_info().devices[i].cc) {
                min_compute_capability = ggml_sycl_info().devices[i].cc;
            }
            if (max_compute_capability < ggml_sycl_info().devices[i].cc) {
                max_compute_capability = ggml_sycl_info().devices[i].cc;
            }
        }
    }

    switch(type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_PTQ1_0:
        case GGML_TYPE_PQ2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
            return max_compute_capability >= VER_GEN9 ? 128 : 64;
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return 64;
        case GGML_TYPE_F16:
        case GGML_TYPE_F32:
            return 1;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            return max_compute_capability >= VER_GEN9 ? 128 : 64;
        case GGML_TYPE_IQ3_S:
            return max_compute_capability >= VER_GEN9 ? 128 : 64;
        case GGML_TYPE_Q6_K:
            return 64;
        default:
            GGML_ABORT("fatal error");
    }
}

static void get_row_split(int64_t * row_low, int64_t * row_high, const ggml_tensor * tensor, const std::array<float, GGML_SYCL_MAX_DEVICES> & tensor_split, int id) {
    const int64_t nrows = ggml_nrows(tensor);
    const int64_t rounding = get_row_rounding(tensor->type, tensor_split);

    *row_low = id == 0 ? 0 : nrows*tensor_split[id];
    *row_low -= *row_low % rounding;
    if (id == ggml_sycl_info().device_count - 1) {
        *row_high = nrows;
    } else {
        *row_high = nrows*tensor_split[id + 1];
        *row_high -= *row_high % rounding;
    }
}

static size_t ggml_nbytes_split(const struct ggml_tensor * tensor, int nrows_split) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return nrows_split*ggml_row_size(tensor->type, tensor->ne[0]);
}

struct ggml_backend_sycl_split_buffer_type_context {
    std::array<float, GGML_SYCL_MAX_DEVICES> tensor_split;
};

struct ggml_backend_sycl_split_buffer_context {
    ~ggml_backend_sycl_split_buffer_context() try {
        for (ggml_tensor_extra_gpu * extra : tensor_extras) {
            release_extra_gpu(extra, streams);
        }
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    std::vector<queue_ptr> streams;
};

static void ggml_backend_sycl_split_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    delete ctx;
}

static void * ggml_backend_sycl_split_buffer_get_base(ggml_backend_buffer_t buffer) {
    // the pointers are stored in the tensor extras, this is just a dummy address and never dereferenced
    return (void *)0x1000;

    GGML_UNUSED(buffer);
}

static enum ggml_status
ggml_backend_sycl_split_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                           ggml_tensor *tensor) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor, "\n").c_str());
    GGML_ASSERT(tensor->view_src == nullptr); // views of split tensors are not supported

    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];

    ggml_tensor_extra_gpu * extra = new ggml_tensor_extra_gpu{};

    ctx->tensor_extras.push_back(extra);
    ctx->streams.push_back(&(dpct::get_current_device().default_queue()));

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        ggml_sycl_set_device(i);
        const queue_ptr stream = ctx->streams[i];
        char * buf;
        SYCL_CHECK(CHECK_TRY_ERROR(buf = (char *)ggml_sycl_malloc_device(size, *stream)));
        if (!buf) {
            char err_buf[1024];
            snprintf(err_buf, 1023, "%s: can't allocate %zu Bytes of memory on device\n", __func__, size);
            throw std::runtime_error(err_buf);
        }
        // set padding to 0 to avoid possible NaN values
        if (size > original_size) {
            /*
            DPCT1009:209: SYCL uses exceptions to report errors and does not use
            the error codes. The original code was commented out and a warning
            string was inserted. You need to rewrite this code.
            */
            SYCL_CHECK(CHECK_TRY_ERROR(
                (*stream)
                    .memset(buf + original_size, 0, size - original_size)
                    .wait()));
        }

        extra->data_device[i] = buf;

        for (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            /*
            DPCT1009:210: SYCL uses exceptions to report errors and does not use
            the error codes. The original code was commented out and a warning
            string was inserted. You need to rewrite this code.
            */
            SYCL_CHECK(
                CHECK_TRY_ERROR(extra->events[i][is] = new sycl::event()));
        }
    }
    tensor->extra = extra;
    return GGML_STATUS_SUCCESS;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void
ggml_backend_sycl_split_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                          ggml_tensor *tensor, const void *data,
                                          size_t offset, size_t size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];
    const size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        const size_t offset_split = row_low*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        const char * buf_host = (const char *)data + offset_split;
        /*
        DPCT1009:211: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_set_device(i);
        const queue_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .memcpy(extra->data_device[i], buf_host, original_size)
                .wait()));
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void
ggml_backend_sycl_split_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                          const ggml_tensor *tensor, void *data,
                                          size_t offset, size_t size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];
    const size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        const size_t offset_split = row_low*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        char * buf_host = (char *)data + offset_split;
        /*
        DPCT1009:212: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_set_device(i);
        const queue_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .memcpy(buf_host, extra->data_device[i], original_size)
                .wait()));
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_split_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
}

static struct ggml_backend_buffer_i ggml_backend_sycl_split_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_sycl_split_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_sycl_split_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_sycl_split_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_sycl_split_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_split_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_sycl_split_buffer_clear,
    /* .reset           = */ NULL,
};

// sycl split buffer type

static const char * ggml_backend_sycl_split_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return GGML_SYCL_NAME "_Split";

    GGML_UNUSED(buft);
}

static bool ggml_backend_buffer_is_sycl_split(ggml_backend_buffer_t buffer) {
   return buffer->buft->iface.get_name == ggml_backend_sycl_split_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_sycl_split_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // since we don't know the exact split after rounding, we cannot allocate the device buffers at this point
    // instead, we allocate them for each tensor separately in init_tensor
    // however, the size still represents the maximum cumulative size of all the device buffers after the tensors are allocated,
    // as returned by get_alloc_size. this limit is enforced during tensor allocation by ggml-alloc, so it must be correct.
    ggml_backend_sycl_split_buffer_context * ctx = new ggml_backend_sycl_split_buffer_context();

    return ggml_backend_buffer_init(buft, ggml_backend_sycl_split_buffer_interface, ctx, size);
}

static size_t ggml_backend_sycl_split_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return SYCL_BUFFER_ALIGNMENT;
    GGML_UNUSED(buft);
}

static size_t ggml_backend_sycl_split_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    ggml_backend_sycl_split_buffer_type_context * ctx = (ggml_backend_sycl_split_buffer_type_context *)buft->context;

    size_t total_size = 0;

    const int64_t ne0 = tensor->ne[0];

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, ctx->tensor_split, i);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        total_size += ggml_nbytes_split(tensor, nrows_split);

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            total_size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return total_size;
}

static bool ggml_backend_sycl_split_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_i ggml_backend_sycl_split_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_sycl_split_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_sycl_split_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_sycl_split_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_sycl_split_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_sycl_split_buffer_type_is_host,
};

ggml_backend_buffer_type_t ggml_backend_sycl_split_buffer_type(const float * tensor_split) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_split_buffer_type\n");
    ggml_check_sycl();
    // FIXME: this is not thread safe
    static std::map<std::array<float, GGML_SYCL_MAX_DEVICES>, struct ggml_backend_buffer_type> buft_map;

    std::array<float, GGML_SYCL_MAX_DEVICES> tensor_split_arr = {};

    bool all_zero = tensor_split == nullptr || std::all_of(tensor_split, tensor_split + GGML_SYCL_MAX_DEVICES, [](float x) { return x == 0.0f; });
    if (all_zero) {
        tensor_split_arr = ggml_sycl_info().default_tensor_split;
    } else {
        float split_sum = 0.0f;
        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            tensor_split_arr[i] = split_sum;
            split_sum += tensor_split[i];
        }
        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            tensor_split_arr[i] /= split_sum;
        }
    }

    auto it = buft_map.find(tensor_split_arr);
    if (it != buft_map.end()) {
        return &it->second;
    }

    struct ggml_backend_buffer_type buft {
        /* .iface   = */ ggml_backend_sycl_split_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), 0),
        /* .context = */ new ggml_backend_sycl_split_buffer_type_context{tensor_split_arr},
    };

    auto result = buft_map.emplace(tensor_split_arr, buft);
    return &result.first->second;
}

// host buffer type

struct ggml_backend_sycl_device_context {
    int device;
    std::string name;
    std::string description;
    int op_offload_min_batch_size;
};

static const char * ggml_backend_sycl_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_SYCL_NAME "_Host";

    GGML_UNUSED(buft);
}

//host pinned memory
static void * ggml_backend_sycl_host_malloc(size_t size) {
    void * ptr = nullptr;
    try {
        ggml_check_sycl();
        // USM host memory is page-locked and device-accessible by construction
        auto & q = dpct::dev_mgr::instance().get_device(0).default_queue();
        ptr = sycl::malloc_host(size, q, sycl::property_list{});
    } catch (...) {
        ptr = nullptr;
    }
    if (ptr == nullptr) {
        GGML_LOG_WARN("%s: failed to allocate %.2f MiB of pinned memory\n", __func__,
                size / 1024.0 / 1024.0);
    }

    return ptr;
}

static void ggml_backend_sycl_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    if (buffer->context == nullptr) {
        return;
    }
    if (g_ggml_sycl_enable_host_pinned_mem) {
        auto & q = dpct::dev_mgr::instance().get_device(0).default_queue();
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(buffer->context, q)));
    } else {
        free_aligned_mem_host((void *) buffer->context);
    }
}

static ggml_backend_buffer_t ggml_backend_sycl_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = g_ggml_sycl_enable_host_pinned_mem ? ggml_backend_sycl_host_malloc(size) :
                                                      aligned_malloc_host(TENSOR_ALIGNMENT, size);
    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    // FIXME: this is a hack to avoid having to implement a new buffer type
    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_sycl_host_buffer_free_buffer;

    return buffer;
}

static size_t ggml_backend_sycl_host_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {

    if (g_ggml_sycl_enable_host_pinned_mem) {
        ggml_backend_sycl_device_context * dev_ctx = (ggml_backend_sycl_device_context *) buft->device->context;
        return dpct::dev_mgr::instance().get_device(dev_ctx->device).get_max_mem_alloc_size();
    } else {
        return SIZE_MAX;
    }
}

ggml_backend_buffer_type_t ggml_backend_sycl_host_buffer_type() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_host_buffer_type\n");
    static struct ggml_backend_buffer_type ggml_backend_sycl_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_sycl_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_sycl_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size     = */ ggml_backend_sycl_host_buffer_type_get_max_size,
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), 0),
        /* .context  = */ nullptr,
    };

    return &ggml_backend_sycl_buffer_type_host;
}

// buffer pool for sycl (legacy)
struct ggml_sycl_pool_leg : public ggml_sycl_pool {
    static const int MAX_SYCL_BUFFERS = 256;

    int device;
    queue_ptr qptr;
    struct ggml_sycl_buffer {
        void * ptr = nullptr;
        size_t size = 0;
    };

    ggml_sycl_buffer buffer_pool[MAX_SYCL_BUFFERS] = {};
    size_t pool_size = 0;

    explicit ggml_sycl_pool_leg(queue_ptr qptr_, int device_) : device(device_), qptr(qptr_) {}

    ~ggml_sycl_pool_leg() {
#ifdef DEBUG_SYCL_POOL
        int    n_cached    = 0;
        size_t bytes_cached = 0;
        for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            if (buffer_pool[i].ptr != nullptr) {
                ++n_cached;
                bytes_cached += buffer_pool[i].size;
            }
        }
        GGML_LOG_INFO("%s: %d buffers, cached = %.2f MiB\n", __func__,
                      n_cached, bytes_cached / 1024.0 / 1024.0);
        const auto slots = format_slots_in_alloc_order();
        if (!slots.empty()) {
            GGML_LOG_INFO("%s: slots MiB: %s\n", __func__, slots.c_str());
        }
#endif

        for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(b.ptr, *qptr)));
                pool_size -= b.size;
            }
        }
        GGML_ASSERT(pool_size == 0);
    }

#ifdef DEBUG_SYCL_POOL
    std::string format_slots_in_alloc_order() const {
        std::string line;
        char buf[32];
        bool first = true;
        for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            if (buffer_pool[i].ptr == nullptr) {
                continue;
            }
            if (!first) {
                line += '/';
            }
            first = false;
            snprintf(buf, sizeof(buf), "%.2f", buffer_pool[i].size / 1024.0 / 1024.0);
            line += buf;
        }
        return line;
    }
#endif

    void * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_sycl_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_sycl_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_sycl_buffer& b = buffer_pool[ibest];
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);

        SYCL_CHECK(CHECK_TRY_ERROR(ptr = (void *)ggml_sycl_malloc_device(look_ahead_size, *qptr)));
        if (!ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on device/GPU\n", __func__, look_ahead_size);
            return nullptr;
        }

        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;

#ifdef DEBUG_SYCL_MALLOC
        GGML_LOG_DEBUG("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, id, nnz,
                (uint32_t)(max_size/1024/1024), (uint32_t)(g_sycl_pool_size[id]/1024/1024), (uint32_t)(size/1024/1024));
#endif

        // GGML_SYCL_DEBUG("ggml_sycl_pool_malloc_leg look_ahead_size=%zu, return %p\n", look_ahead_size, ptr);
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        GGML_LOG_WARN("WARNING: sycl buffer pool full, increase MAX_sycl_BUFFERS\n");
        SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(ptr, *qptr)));
        pool_size -= size;
    }
};

// pool with virtual memory management
#if defined(GGML_SYCL_SUPPORT_VMM)
struct ggml_sycl_pool_vmm : public ggml_sycl_pool {
    static const size_t SYCL_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    int           device;
    sycl::context ctx;
    sycl::device  dev;

    uintptr_t pool_addr = 0;
    size_t    pool_used = 0;
    size_t    pool_size = 0;
    size_t    granularity;

    // physical_mem owns the commits (unlike cuMemMap)
    struct mapping {
        sycl::ext::oneapi::experimental::physical_mem phys;
        void * map_ptr;
    };
    std::vector<mapping> mappings;

    explicit ggml_sycl_pool_vmm(queue_ptr qptr_, int device_) :
        device(device_),
        ctx(qptr_->get_context()),
        dev(qptr_->get_device()),
        granularity(ggml_sycl_info().devices[device_].vmm_granularity) {
    }

    ~ggml_sycl_pool_vmm() {
        if (pool_addr == 0) {
            return;
        }

        // Per spec, unmap must (a) match the exact (ptr, size) of an earlier
        // physical_mem::map() call and (b) precede destruction of the
        // physical_mem objects (their dtors won't unmap).
        for (auto & m : mappings) {
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::ext::oneapi::experimental::unmap(
                m.map_ptr, m.phys.size(), ctx)));
        }
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::ext::oneapi::experimental::free_virtual_mem(
            pool_addr, SYCL_POOL_VMM_MAX_SIZE, ctx)));
    }

    void * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        size = GGML_PAD(size, SYCL_BUFFER_ALIGNMENT);

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = GGML_PAD(size - avail, granularity);

            GGML_ASSERT(pool_size + reserve_size <= SYCL_POOL_VMM_MAX_SIZE);

            // allocate more physical memory
            std::optional<sycl::ext::oneapi::experimental::physical_mem> phys;
            SYCL_CHECK(CHECK_TRY_ERROR(phys.emplace(dev, ctx, reserve_size)));

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                SYCL_CHECK(CHECK_TRY_ERROR(
                    pool_addr = sycl::ext::oneapi::experimental::reserve_virtual_mem(
                        SYCL_POOL_VMM_MAX_SIZE, ctx)));
            }

            // map at the end of the pool
            void * map_ptr = nullptr;
            SYCL_CHECK(CHECK_TRY_ERROR(
                map_ptr = phys->map(pool_addr + pool_size, reserve_size,
                                    sycl::ext::oneapi::experimental::address_access_mode::read_write)));

            // stash these so we could unmap this exact range in dtor
            mappings.push_back({
                std::move(*phys),
                map_ptr,
            });

            // add to the pool
            pool_size += reserve_size;

#ifdef DEBUG_SYCL_MALLOC
            GGML_LOG_INFO("sycl pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
                          device, (unsigned long long) (pool_size/1024/1024),
                          (unsigned long long) (reserve_size/1024/1024));
#endif
        }

        GGML_ASSERT(pool_addr != 0);

        void * ptr = reinterpret_cast<void *>(pool_addr + pool_used);
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_SYCL_MALLOC
        GGML_LOG_INFO("sycl pool[%d]: allocated %llu bytes at %p\n", device, (unsigned long long) size, ptr);
#endif

        return ptr;
    }

    void free(void * ptr, size_t size) override {
#ifdef DEBUG_SYCL_MALLOC
        GGML_LOG_INFO("sycl pool[%d]: freed %llu bytes at %p\n", device, (unsigned long long) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == reinterpret_cast<void *>(pool_addr + pool_used));
    }
};
#endif // defined(GGML_SYCL_SUPPORT_VMM)

struct ggml_sycl_pool_host : public ggml_sycl_pool {
    queue_ptr qptr;
    int       device;

    inline static int counter{ 0 };

    struct ggml_sycl_buffer {
        void * ptr  = nullptr;
        size_t size = 0;
    };

    // Set arbitrarly to 64
    static constexpr int          MAX_POOL_SIZE{ 64 };
    std::vector<ggml_sycl_buffer> buffer_pool = std::vector<ggml_sycl_buffer>(MAX_POOL_SIZE);
    size_t                        pool_size   = 0;

    explicit ggml_sycl_pool_host(queue_ptr qptr_, int device_) : qptr(qptr_), device(device_) {}

    ~ggml_sycl_pool_host() {
        for (int i = 0; i < MAX_POOL_SIZE; ++i) {
            ggml_sycl_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(b.ptr, *qptr)));
                b.ptr = nullptr;
                pool_size -= b.size;
                b.size = 0;
            }
        }
        counter = 0;
    }

    void * alloc(size_t size, size_t * actual_size) override {
        if (counter == MAX_POOL_SIZE) {
            ggml_sycl_buffer b               = buffer_pool[0];
            void *           ptr             = b.ptr;
            *actual_size                     = b.size;
            counter                          = 1;
            return ptr;
        }
        ggml_sycl_buffer & b = buffer_pool[counter];

        if (b.ptr == nullptr) {
            void * ptr;

            SYCL_CHECK(CHECK_TRY_ERROR(ptr = (void *) sycl::malloc_host(size, *qptr)));
            if (!ptr) {
                GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
                return nullptr;
            }
            pool_size += size;
            *actual_size = size;
            counter      = counter + 1;
            return ptr;
        } else {
            ++counter;
            b.size = size;
            return b.ptr;
        }
    }

    void free(void * ptr, size_t size) override {
        // if the pool is not completed add the pointer to it in place of the first nullptr found.
        // Otherwise do nothing, pointers will be freed once the pool is deallocated.
        for (int i = 0; i < MAX_POOL_SIZE; ++i) {
            ggml_sycl_buffer & b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr  = ptr;
                b.size = size;
                return;
            }
        }
    }
};

std::unique_ptr<ggml_sycl_pool> ggml_backend_sycl_context::new_pool_for_host(queue_ptr qptr, int device) {
    // return pool for the host to speed up memory management
    return std::unique_ptr<ggml_sycl_pool>(new ggml_sycl_pool_host(qptr, device));
}

std::unique_ptr<ggml_sycl_pool> ggml_backend_sycl_context::new_pool_for_device(queue_ptr qptr, int device) {
#if defined(GGML_SYCL_SUPPORT_VMM)
    if (g_ggml_sycl_enable_vmm && ggml_sycl_info().devices[device].vmm) {
        return std::unique_ptr<ggml_sycl_pool>(new ggml_sycl_pool_vmm(qptr, device));
    }
#endif // defined(GGML_SYCL_SUPPORT_VMM)
    return std::unique_ptr<ggml_sycl_pool>(new ggml_sycl_pool_leg(qptr, device));
}


std::unique_ptr<ggml_sycl_fattn_kv_buffers> ggml_backend_sycl_context::new_fattn_kv_buffers(queue_ptr qptr, int device) {
    return std::unique_ptr<ggml_sycl_fattn_kv_buffers>(new ggml_sycl_fattn_kv_buffers(qptr, device));
}

/// kernels
typedef void (*ggml_sycl_op_mul_mat_t)(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const queue_ptr &stream);



static void mul_mat_p021_f16_f32(
    const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int nchannels_x, const int nchannels_y,
    const sycl::nd_item<3> &item_ct1) {

    const sycl::half *x = (const sycl::half *)vx;

    const int row_x = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                      item_ct1.get_local_id(1);
    const int channel = item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                        item_ct1.get_local_id(0);
    const int channel_x = channel / (nchannels_y / nchannels_x);

    const int nrows_y = ncols_x;
    const int nrows_dst = nrows_x;
    const int row_dst = row_x;

    float tmp = 0.0f;

    for (int col_x0 = 0; col_x0 < ncols_x;
         col_x0 += item_ct1.get_local_range(2)) {
        const int col_x = col_x0 + item_ct1.get_local_id(2);

        if (col_x >= ncols_x) {
            break;
        }

        // x is transposed and permuted
        const int ix = row_x*nchannels_x*ncols_x + channel_x*ncols_x + col_x;
        const float xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<float, sycl::rounding_mode::automatic>()[0];

        const int row_y = col_x;


        // y is not transposed but permuted
        const int iy = channel*nrows_y + row_y;

        tmp += xi * y[iy];
    }

    // dst is not transposed and not permuted
    const int idst = channel*nrows_dst + row_dst;

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

static void mul_mat_vec_nc_f16_f32( // nc == non-contiguous
    const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst, const int ncols_x, const int nrows_x,
    const int row_stride_x, const int channel_stride_x,const int channel_stride_y, const int channel_x_divisor,
    const sycl::nd_item<3> &item_ct1) {

    const sycl::half *x = (const sycl::half *)vx;

    const int row_x = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                      item_ct1.get_local_id(1);
    const int channel = item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                        item_ct1.get_local_id(0);
    const int channel_x = channel / channel_x_divisor;

    const int nrows_dst = nrows_x;
    const int row_dst   = row_x;

    const int idst = channel*nrows_dst + row_dst;

    float tmp = 0.0f;

    for (int col_x0 = 0; col_x0 < ncols_x;
         col_x0 += item_ct1.get_local_range(2)) {
        const int col_x = col_x0 + item_ct1.get_local_id(2);

        if (col_x >= ncols_x) {
            break;
        }

        const int row_y = col_x;

        const int ix = channel_x*channel_stride_x + row_x*row_stride_x + col_x;
        const int iy = channel * channel_stride_y + row_y;

        const float xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<float, sycl::rounding_mode::automatic>()[0];

        tmp += xi * y[iy];
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

static void k_sum_rows_f32(const float * x, float * dst, const int ncols,
                           const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(1);
    const int col = item_ct1.get_local_id(2);

    float sum = 0.0f;
    for (int i = col; i < ncols; i += item_ct1.get_local_range(2)) {
        sum += x[row * ncols + i];
    }

    sum = warp_reduce_sum(sum, item_ct1);

    if (col == 0) {
        dst[row] = sum;
    }
}


template<typename T>
static inline void ggml_sycl_swap(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template <ggml_sort_order order>
__dpct_inline__ static void
k_argsort_f32_i32(const float *x, int *dst, const int ncols, int ncols_pad,
                  const int tasks_per_thread, const sycl::nd_item<3> &item_ct1,
                  uint8_t *dpct_local) {
    // bitonic sort
    int col_index =  item_ct1.get_local_id(2);
    int row = item_ct1.get_group(1);

    for (int i = 0; i < tasks_per_thread; i++) {
        int col = col_index * tasks_per_thread + i;
        if (col >= ncols_pad) {
            return;
        }
    }

    const float * x_row = x + row * ncols;
    auto dst_row = (int *)dpct_local;

    // initialize indices
    for (int i=0;i<tasks_per_thread;i++){
        int col = col_index*tasks_per_thread+i;
        dst_row[col] = col;
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int i = 0; i < tasks_per_thread; i++) {
                int col = col_index * tasks_per_thread + i;
                int ixj = col ^ j;
                if (ixj > col) {
                    if ((col & k) == 0) {
                        if (dst_row[col] >= ncols ||
                            (dst_row[ixj] < ncols &&
                             (order == GGML_SORT_ORDER_ASC
                                  ? x_row[dst_row[col]] > x_row[dst_row[ixj]]
                                  : x_row[dst_row[col]] <
                                        x_row[dst_row[ixj]]))) {
                            ggml_sycl_swap(dst_row[col], dst_row[ixj]);
                        }
                    } else {
                        if (dst_row[ixj] >= ncols ||
                            (dst_row[col] < ncols &&
                             (order == GGML_SORT_ORDER_ASC
                                  ? x_row[dst_row[col]] < x_row[dst_row[ixj]]
                                  : x_row[dst_row[col]] >
                                        x_row[dst_row[ixj]]))) {
                            ggml_sycl_swap(dst_row[col], dst_row[ixj]);
                        }
                    }
                }
                item_ct1.barrier(sycl::access::fence_space::local_space);
            }
        }
    }

    // copy the result to dst without the padding
    for (int i = 0; i < tasks_per_thread; i++) {
        int col = col_index * tasks_per_thread + i;
        if (col < ncols) {
            dst[row * ncols + col] = dst_row[col];
        }
    }
}

static void diag_mask_inf_f32(const float * x, float * dst, const int ncols, const int rows_per_channel, const int n_past,
                              const sycl::nd_item<3> &item_ct1) {
    const int col = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1);
    const int row = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);

    if (col >= ncols) {
        return;
    }

    const int i = row*ncols + col;
    //dst[i] = col > (n_past + row % rows_per_channel) ? -INFINITY : x[i];
    //dst[i] = x[i] - (col > n_past + row % rows_per_channel) * INT_MAX; // equivalent within rounding error but slightly faster on GPU
    dst[i] = x[i] - (col > n_past + row % rows_per_channel) * FLT_MAX;
}

static void scale_f32(const float * x, float * dst, const float scale, const float bias, const int k,
                      const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }

    dst[i] = scale * x[i] + bias;
}


static void ggml_mul_mat_p021_f16_f32_sycl(const void *vx, const float *y,
                                           float *dst, const int ncols_x,
                                           const int nrows_x,
                                           const int nchannels_x,
                                           const int nchannels_y,
                                           queue_ptr stream) {

    const sycl::range<3> block_nums(nchannels_y, nrows_x, 1);
    const sycl::range<3> block_dims(1, 1, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_p021_f16_f32(vx, y, dst, ncols_x, nrows_x, nchannels_x,
                                     nchannels_y, item_ct1);
            });
    }
}

static void ggml_mul_mat_vec_nc_f16_f32_sycl(
    const void *vx, const float *y, float *dst, const int ncols_x,
    const int nrows_x, const int row_stride_x, const int nchannels_x,
    const int nchannels_y, const int channel_stride_x, const int channel_stride_y, queue_ptr stream) {

    const sycl::range<3> block_nums(nchannels_y, nrows_x, 1);
    const sycl::range<3> block_dims(1, 1, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_nc_f16_f32(vx, y, dst, ncols_x, nrows_x,
                                       row_stride_x, channel_stride_x, channel_stride_y,
                                       nchannels_y / nchannels_x, item_ct1);
            });
    }
}



static void scale_f32_sycl(const float *x, float *dst, const float scale, const float bias,
                           const int k, queue_ptr stream) {
    const int num_blocks = (k + SYCL_SCALE_BLOCK_SIZE - 1) / SYCL_SCALE_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_SCALE_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_SCALE_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            scale_f32(x, dst, scale, bias, k, item_ct1);
        });
}


static void sum_rows_f32_sycl(const float *x, float *dst, const int ncols,
                              const int nrows, queue_ptr stream) {
    const sycl::range<3> block_dims(1, 1, WARP_SIZE);
    const sycl::range<3> block_nums(1, nrows, 1);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1)
                             [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 k_sum_rows_f32(x, dst, ncols, item_ct1);
                             });
}

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

static void init_argsort_indices_padded(
        int * idx,
        const int nrows,
        const int ncols_pad,
        const sycl::nd_item<1> & item_ct1) {
    const size_t gid = item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);
    const size_t total = (size_t) nrows * (size_t) ncols_pad;

    if (gid >= total) {
        return;
    }

    idx[gid] = (int) (gid % (size_t) ncols_pad);
}

template <ggml_sort_order order>
static void argsort_f32_i32_global_pass(const float *            x,
                                        int *                    idx,
                                        const int                ncols,
                                        const int                nrows,
                                        const int                ncols_pad,
                                        const int                j,
                                        const int                k,
                                        const sycl::nd_item<1> & item_ct1) {
    const size_t gid   = item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);
    const size_t total = (size_t) nrows * (size_t) ncols_pad;

    if (gid >= total) {
        return;
    }

    const int row = (int) (gid / (size_t) ncols_pad);
    const int col = (int) (gid % (size_t) ncols_pad);
    const int ixj = col ^ j;

    if (ixj <= col || ixj >= ncols_pad) {
        return;
    }

    const size_t base  = (size_t) row * (size_t) ncols_pad;
    const size_t pos_a = base + (size_t) col;
    const size_t pos_b = base + (size_t) ixj;

    const int a = idx[pos_a];
    const int b = idx[pos_b];

    bool do_swap = false;

    if ((col & k) == 0) {
        if (a >= ncols ||
            (b < ncols &&
             (order == GGML_SORT_ORDER_ASC ?
                  x[(size_t) row * (size_t) ncols + (size_t) a] > x[(size_t) row * (size_t) ncols + (size_t) b] :
                  x[(size_t) row * (size_t) ncols + (size_t) a] < x[(size_t) row * (size_t) ncols + (size_t) b]))) {
            do_swap = true;
        }
    } else {
        if (b >= ncols ||
            (a < ncols &&
             (order == GGML_SORT_ORDER_ASC ?
                  x[(size_t) row * (size_t) ncols + (size_t) a] < x[(size_t) row * (size_t) ncols + (size_t) b] :
                  x[(size_t) row * (size_t) ncols + (size_t) a] > x[(size_t) row * (size_t) ncols + (size_t) b]))) {
            do_swap = true;
        }
    }

    if (do_swap) {
        idx[pos_a] = b;
        idx[pos_b] = a;
    }
}

static void copy_argsort_indices_unpadded(const int *              idx_padded,
                                          int *                    dst,
                                          const int                nrows,
                                          const int                ncols,
                                          const int                ncols_pad,
                                          const sycl::nd_item<1> & item_ct1) {
    const size_t gid   = item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);
    const size_t total = (size_t) nrows * (size_t) ncols;

    if (gid >= total) {
        return;
    }

    const int row = (int) (gid / (size_t) ncols);
    const int col = (int) (gid % (size_t) ncols);

    dst[(size_t) row * (size_t) ncols + (size_t) col] = idx_padded[(size_t) row * (size_t) ncols_pad + (size_t) col];
}

static void argsort_f32_i32_sycl(const float *x, int *dst, const int ncols,
                                 const int nrows, ggml_sort_order order,
                                 queue_ptr stream, int device, ggml_sycl_pool & pool) {
    // bitonic sort requires ncols to be power of 2
    const int ncols_pad = next_power_of_2(ncols);
    const size_t shared_mem = (size_t) ncols_pad * sizeof(int);
    const size_t smpbo = ggml_sycl_info().devices[device].smpbo;

    if (shared_mem > smpbo) {
        ggml_sycl_pool_alloc<int> idx_padded_alloc(pool, (size_t) nrows * (size_t) ncols_pad);
        int *                     idx_padded = idx_padded_alloc.get();

        constexpr size_t block_size     = 256;
        const size_t     total_padded   = (size_t) nrows * (size_t) ncols_pad;
        const size_t     nblocks_padded = (total_padded + block_size - 1) / block_size;

        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblocks_padded * block_size), sycl::range<1>(block_size)),
            [=](sycl::nd_item<1> item_ct1) { init_argsort_indices_padded(idx_padded, nrows, ncols_pad, item_ct1); });

        for (int k = 2; k <= ncols_pad; k *= 2) {
            for (int j = k / 2; j > 0; j /= 2) {
                if (order == GGML_SORT_ORDER_ASC) {
                    stream->parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(nblocks_padded * block_size), sycl::range<1>(block_size)),
                        [=](sycl::nd_item<1> item_ct1) {
                            argsort_f32_i32_global_pass<GGML_SORT_ORDER_ASC>(x, idx_padded, ncols, nrows, ncols_pad, j,
                                                                             k, item_ct1);
                        });
                } else if (order == GGML_SORT_ORDER_DESC) {
                    stream->parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(nblocks_padded * block_size), sycl::range<1>(block_size)),
                        [=](sycl::nd_item<1> item_ct1) {
                            argsort_f32_i32_global_pass<GGML_SORT_ORDER_DESC>(x, idx_padded, ncols, nrows, ncols_pad, j,
                                                                              k, item_ct1);
                        });
                } else {
                    GGML_ABORT("invalid sort order");
                }
            }
        }

        const size_t total   = (size_t) nrows * (size_t) ncols;
        const size_t nblocks = (total + block_size - 1) / block_size;
        stream->parallel_for(sycl::nd_range<1>(sycl::range<1>(nblocks * block_size), sycl::range<1>(block_size)),
                             [=](sycl::nd_item<1> item_ct1) {
                                 copy_argsort_indices_unpadded(idx_padded, dst, nrows, ncols, ncols_pad, item_ct1);
                             });

        return;
    }

    int nth = 1;
    int max_block_size = ggml_sycl_info().max_work_group_sizes[device];
    while (nth < ncols_pad && nth < max_block_size)
        nth *= 2;
    if (nth > max_block_size)
        nth = max_block_size;

    const int tasks_per_thread = ncols_pad / nth;

    const sycl::range<3> block_dims(1, 1, nth);
    const sycl::range<3> block_nums(1, nrows, 1);

    if (order == GGML_SORT_ORDER_ASC) {
        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(shared_mem), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    k_argsort_f32_i32<GGML_SORT_ORDER_ASC>(
                        x, dst, ncols, ncols_pad, tasks_per_thread, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    } else if (order == GGML_SORT_ORDER_DESC) {
        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(shared_mem), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    k_argsort_f32_i32<GGML_SORT_ORDER_DESC>(
                        x, dst, ncols, ncols_pad, tasks_per_thread, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    } else {
        GGML_ABORT("fatal error");
    }
}

static void top_k_f32_sycl(
    const float * src,
    int32_t * dst_indices,
    const int64_t ncols,
    const int64_t nrows,
    const int k,
    dpct::queue_ptr main_stream
) {
    const int block_size = 128;

    const sycl::range<1> block_dims(block_size);
    const sycl::range<1> grid_dims(nrows);

    main_stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_vals(sycl::range<1>(block_size * k), cgh);
        sycl::local_accessor<int, 1> shared_idx(sycl::range<1>(block_size * k), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(grid_dims * block_dims, block_dims),
            [=](sycl::nd_item<1> item_ct1) {
                const int row = item_ct1.get_group(0);
                const int tid = item_ct1.get_local_id(0);

                if (row >= nrows) return;

                const float * src_row = src + row * ncols;
                int32_t * dst_idx_row = dst_indices + row * k;

                float local_vals[32];
                int local_idx[32];

                for (int i = 0; i < k; i++) {
                    local_vals[i] = -FLT_MAX;
                    local_idx[i] = -1;
                }

                for (int col = tid; col < ncols; col += block_size) {
                    float val = src_row[col];

                    if (val > local_vals[k-1]) {
                        int pos = k - 1;
                        while (pos > 0 && val > local_vals[pos - 1]) {
                            pos--;
                        }

                        for (int i = k - 1; i > pos; i--) {
                            local_vals[i] = local_vals[i - 1];
                            local_idx[i] = local_idx[i - 1];
                        }
                        local_vals[pos] = val;
                        local_idx[pos] = col;
                    }
                }

                for (int i = 0; i < k; i++) {
                    shared_vals[tid * k + i] = local_vals[i];
                    shared_idx[tid * k + i] = local_idx[i];
                }
                item_ct1.barrier(sycl::access::fence_space::local_space);

                if (tid == 0) {
                    float final_vals[32];
                    int final_idx[32];

                    for (int i = 0; i < k; i++) {
                        final_vals[i] = -FLT_MAX;
                        final_idx[i] = -1;
                    }

                    for (int t = 0; t < block_size; t++) {
                        for (int i = 0; i < k; i++) {
                            float val = shared_vals[t * k + i];
                            int idx = shared_idx[t * k + i];

                            if (val > final_vals[k-1]) {
                                int pos = k - 1;
                                while (pos > 0 && val > final_vals[pos - 1]) {
                                    pos--;
                                }

                                for (int j = k - 1; j > pos; j--) {
                                    final_vals[j] = final_vals[j - 1];
                                    final_idx[j] = final_idx[j - 1];
                                }
                                final_vals[pos] = val;
                                final_idx[pos] = idx;
                            }
                        }
                    }

                    for (int i = 0; i < k; i++) {
                        dst_idx_row[i] = final_idx[i];
                    }

                    if (k > 1) {
                        int32_t temp = dst_idx_row[0];
                        dst_idx_row[0] = dst_idx_row[1];
                        dst_idx_row[1] = temp;
                    }
                }
            });
    });
}

static void argmax_f32_i32_sycl(const float *x, int *dst, const int ncols,
                               const int nrows, queue_ptr stream) {
    const sycl::range<3> block_dims(1, 1, SYCL_ARGMAX_BLOCK_SIZE);
    const sycl::range<3> block_nums(1, nrows, 1);
    const size_t shared_mem = 256 * sizeof(float);

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_data(
            sycl::range<1>(shared_mem/sizeof(float)), cgh);
        sycl::local_accessor<int, 1> shared_indices(
            sycl::range<1>(shared_mem/sizeof(float)), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                const int tid = item_ct1.get_local_id(2);
                const int row = item_ct1.get_global_id(1);

                float max_val = -INFINITY;
                int max_idx = -1;

                for (int col = tid; col < ncols; col += 256) {
                    float val = x[row * ncols + col];
                    if (val > max_val) {
                        max_val = val;
                        max_idx = col;
                    }
                }

                shared_data[tid] = max_val;
                shared_indices[tid] = max_idx;
                item_ct1.barrier(sycl::access::fence_space::local_space);

                for (int stride = 256/2; stride > 0; stride >>= 1) {
                    if (tid < stride) {
                        float val1 = shared_data[tid];
                        float val2 = shared_data[tid + stride];
                        if (val2 > val1) {
                            shared_data[tid] = val2;
                            shared_indices[tid] = shared_indices[tid + stride];
                        }
                    }
                    item_ct1.barrier(sycl::access::fence_space::local_space);
                }


                if (tid == 0) {
                    dst[row] = shared_indices[0];
                }
            });
    });
}
static void diag_mask_inf_f32_sycl(const float *x, float *dst,
                                   const int ncols_x, const int nrows_x,
                                   const int rows_per_channel, const int n_past,
                                   queue_ptr stream) {
    const sycl::range<3> block_dims(1, SYCL_DIAG_MASK_INF_BLOCK_SIZE, 1);
    const int block_num_x = (ncols_x + SYCL_DIAG_MASK_INF_BLOCK_SIZE - 1) / SYCL_DIAG_MASK_INF_BLOCK_SIZE;
    const sycl::range<3> block_nums(1, block_num_x, nrows_x);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             diag_mask_inf_f32(x, dst, ncols_x,
                                               rows_per_channel, n_past,
                                               item_ct1);
                         });
}

static dpct::err0 ggml_sycl_cpy_tensor_2d(void *dst,
                                          const struct ggml_tensor *src,
                                          int64_t i3, int64_t i2,
                                          int64_t i1_low, int64_t i1_high,
                                          queue_ptr stream) try {

    dpct::memcpy_direction kind;
    char * src_ptr;
    if (ggml_backend_buffer_is_host(src->buffer)) {
        kind = dpct::host_to_device;
        //GGML_SYCL_DEBUG("%s: Host buffer type src tensor\n", __func__);
        src_ptr = (char *) src->data;
        // GGML_SYCL_DEBUG("ggml_sycl_cpy_tensor_2d  GGML_BACKEND_TYPE_CPU src_ptr %p\n", src_ptr);
    } else if (ggml_backend_buffer_is_sycl(src->buffer)) {
        // If buffer is a SYCL buffer
        //GGML_SYCL_DEBUG("%s: SYCL buffer type src tensor\n", __func__);
        kind    = dpct::device_to_device;
        src_ptr = (char *) src->data;
    } else if (ggml_backend_buffer_is_sycl_split(src->buffer)) {
        /*
        If buffer is a SYCL split buffer
        */
        //GGML_SYCL_DEBUG("%s: Split buffer type src tensor\n", __func__);
        GGML_ASSERT(i1_low == 0 && i1_high == src->ne[1]);
        kind = dpct::device_to_device;
        ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) src->extra;
        int id;
        SYCL_CHECK(CHECK_TRY_ERROR(
            id = get_current_device_id()));
        // GGML_SYCL_DEBUG("current device index %d\n", id);
        src_ptr = (char *) extra->data_device[id];
    } else {
        // GGML_SYCL_DEBUG("GGML_ABORT("fatal error")\n");
        GGML_ABORT("fatal error");
    }
    char * dst_ptr = (char *) dst;

    GGML_TENSOR_LOCALS_1(int64_t, ne, src, ne);
    GGML_TENSOR_LOCALS(int64_t, nb, src, nb);
    const enum ggml_type type = src->type;
    const int64_t ts = ggml_type_size(type);
    const int64_t bs = ggml_blck_size(type);
    int64_t i1_diff = i1_high - i1_low;

    const char * x = src_ptr + i1_low*nb1 + i2*nb2 + i3*nb3;
    if (nb0 == ts && nb1 == ts*ne0/bs) {
        // GGML_SYCL_DEBUG("stream->memcpy: dst_ptr=%p, x=%p, size=%lu\n", dst_ptr, x, i1_diff * nb1);
        // return CHECK_TRY_ERROR(stream->memcpy(dst_ptr, x, i1_diff * nb1));
        return CHECK_TRY_ERROR(dpct::async_dpct_memcpy(dst_ptr, x, i1_diff * nb1,
                                    kind, *stream));

    } else if (nb0 == ts) {
        return CHECK_TRY_ERROR(
            dpct::async_dpct_memcpy(dst_ptr, ts * ne0 / bs, x, nb1,
                                    ts * ne0 / bs, i1_diff, kind, *stream));
    } else {
        for (int64_t i1 = 0; i1 < i1_diff; i1++) {
            const void * rx = (const void *) ((const char *) x + i1*nb1);
            void * rd = (void *) (dst_ptr + i1*ts*ne0/bs);
            // pretend the row is a matrix with cols=1
            dpct::err0 r = CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
                rd, ts / bs, rx, nb0, ts / bs, ne0, kind, *stream));
            /*
            DPCT1001:85: The statement could not be removed.
            */
            /*
            DPCT1000:86: Error handling if-stmt was detected but could not be
            rewritten.
            */
            if (r != 0) return r;
        }
        return 0;
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

inline void ggml_sycl_op_mul_mat_sycl(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const queue_ptr &stream) try {

    GGML_ASSERT(src0_dd_i  != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_dd_i   != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne00 == ne10);

    const int64_t row_diff = row_high - row_low;

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));

    const int64_t ne0 = dst->ne[0]; // used by MKL only
    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    int ldc = id == ctx.device ? ne0 : row_diff; // used by MKL only

#ifdef GGML_SYCL_F16
    bool use_fp16 = true;  // TODO(Yu) SYCL capability check
#else
    // ternary weights are exact in FP16 and the FP16 GEMM is much faster than FP32; other quantized weights
    // take it for large batches too (ARC-LAB: fp16 GEMM runs on XMX, fp32 GEMM is ALU bound)
    bool use_fp16 = src0->type == GGML_TYPE_PQ2_0 || src0->type == GGML_TYPE_PTQ1_0 ||
                    (g_ggml_sycl_quant_f16_gemm && ggml_is_quantized(src0->type) && src1_ncols > 32);
#endif

#if GGML_SYCL_DNNL
    // PQ2_0 / PTQ1_0 batches: repack to s4 + group scales and let oneDNN decompress inside the GEMM (XMX),
    // instead of writing and reading back a full fp16 copy of the weights.
    if (g_ggml_sycl_quant_w8a8 && g_ggml_sycl_enable_dnn && (src0->type == GGML_TYPE_PQ2_0 || src0->type == GGML_TYPE_PTQ1_0) &&
        src1_ncols > 32 && ggml_is_contiguous(src0) && row_diff == src0->ne[1] && ldc == row_diff &&
        dst->op_params[0] == GGML_PREC_DEFAULT) {
        const bool reordered = src0->extra && ((ggml_tensor_extra_gpu *) src0->extra)->optimized_feature.reorder;
        if (ggml_sycl_w8a8_mul_mat(ctx, src0->type, reordered, src0_dd_i, src1_ddf_i, dst_dd_i, row_diff, src1_ncols,
                                   ne10, stream)) {
            return;
        }
    }
    if (g_ggml_sycl_quant_s4_gemm && g_ggml_sycl_enable_dnn && (src0->type == GGML_TYPE_PQ2_0 || src0->type == GGML_TYPE_PTQ1_0) &&
        src1_ncols > 32 && ggml_is_contiguous(src0) && row_diff == src0->ne[1] && ldc == row_diff &&
        dst->op_params[0] == GGML_PREC_DEFAULT) {
        const bool reordered = src0->extra && ((ggml_tensor_extra_gpu *) src0->extra)->optimized_feature.reorder;
        ggml_sycl_pool_alloc<uint8_t>    w4(ctx.pool(), row_diff * ne00 / 2);
        ggml_sycl_pool_alloc<float>      scales(ctx.pool(), row_diff * ne00 / 128);
        ggml_sycl_quant_to_s4(src0->type, reordered, src0_dd_i, w4.get(), scales.get(), row_diff, ne00, stream);
        ggml_sycl_pool_alloc<sycl::half> src1_as_f16(ctx.pool(), src1_ncols * ne10);
        ggml_get_to_fp16_sycl(src1->type, dst)(src1_ddf_i, src1_as_f16.get(), src1_ncols * ne10, stream);
        DnnlGemmWrapper::gemm_s4_group(ctx, row_diff, src1_ncols, ne10, src1_as_f16.get(), w4.get(), scales.get(), dst_dd_i, stream);
        return;
    }
#endif

#if GGML_SYCL_DNNL && defined(GGML_SYCL_HAS_BF16)
    // Fast path for bf16 src0
    if (src0->type == GGML_TYPE_BF16 && g_ggml_sycl_enable_dnn && ggml_is_contiguous(src0) &&
        row_diff == src0->ne[1]) {
        using bf16_t = sycl::ext::oneapi::bfloat16;
        ggml_sycl_pool_alloc<bf16_t> src1_as_bf16(ctx.pool(), src1_ncols*ne10);
        if (src1->type != GGML_TYPE_BF16) {
            const to_bf16_sycl_t to_bf16_sycl = ggml_get_to_bf16_sycl(src1->type, dst);
            GGML_ASSERT(to_bf16_sycl != nullptr);
            to_bf16_sycl(src1_ddf_i, src1_as_bf16.get(), src1_ncols*ne10, stream);
        } else {
            stream->memcpy(src1_as_bf16.get(), src1_ddf_i, src1_ncols*ne10*sizeof(bf16_t));
        }
        DnnlGemmWrapper::row_gemm(ctx, row_diff, src1_ncols, ne10,
                                  src0_dd_i, DnnlGemmWrapper::to_dt<bf16_t>(),
                                  src1_as_bf16.get(), DnnlGemmWrapper::to_dt<bf16_t>(),
                                  dst_dd_i, DnnlGemmWrapper::to_dt<float>(), stream);
        GGML_UNUSED(dst);
        GGML_UNUSED(src1_ddq_i);
        GGML_UNUSED(src1_padded_row_size);
        return;
    }
#endif

    if ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && use_fp16 && ggml_is_contiguous(src0) &&
        row_diff == src0->ne[1] && dst->op_params[0] == GGML_PREC_DEFAULT) {
        ggml_sycl_pool_alloc<sycl::half> src0_as_f16(ctx.pool());
        if (src0->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp16");
            const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src0->type, dst);
            GGML_ASSERT(to_fp16_sycl != nullptr);
            size_t ne = row_diff*ne00;
            src0_as_f16.alloc(ne);
            to_fp16_sycl(src0_dd_i, src0_as_f16.get(), ne, stream);
        }
        const sycl::half *src0_ptr = src0->type == GGML_TYPE_F16
                                         ? (const sycl::half *)src0_dd_i
                                         : src0_as_f16.get();

        ggml_sycl_pool_alloc<sycl::half> src1_as_f16(ctx.pool());
        if (src1->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp16");
            const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type, dst);
            GGML_ASSERT(to_fp16_sycl != nullptr);
            size_t ne = src1_ncols*ne10;
            src1_as_f16.alloc(ne);
            to_fp16_sycl(src1_ddf_i, src1_as_f16.get(), ne, stream);
        }
        const sycl::half *src1_ptr = src1->type == GGML_TYPE_F16
                ? (const sycl::half *)src1->data + src1_padded_row_size
                                         : src1_as_f16.get();

#if GGML_SYCL_DNNL
        if (g_ggml_sycl_enable_dnn) {
                DnnlGemmWrapper::row_gemm(ctx,row_diff, src1_ncols , ne10, src0_ptr,
                                     DnnlGemmWrapper::to_dt<sycl::half>(), src1_ptr, DnnlGemmWrapper::to_dt<sycl::half>(),
                                      dst_dd_i, DnnlGemmWrapper::to_dt<float>(), stream);
        }
        else
#endif
        {
            const float alpha = 1.0f;
            const float beta  = 0.0f;
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm(
                *stream, oneapi::mkl::transpose::trans,
                oneapi::mkl::transpose::nontrans, row_diff, src1_ncols, ne10,
                &alpha, src0_ptr, dpct::library_data_t::real_half, ne00,
                src1_ptr, dpct::library_data_t::real_half, ne10, &beta,
                dst_dd_i, dpct::library_data_t::real_float, ldc,
                dpct::library_data_t::real_float)));
        }
    } else {
        ggml_sycl_pool_alloc<float> src0_ddq_as_f32(ctx.pool());
        ggml_sycl_pool_alloc<float> src1_ddq_as_f32(ctx.pool());
        if (src0->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp32");
            const to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(src0->type, dst);
            GGML_ASSERT(to_fp32_sycl != nullptr);
            src0_ddq_as_f32.alloc(row_diff*ne00);
            to_fp32_sycl(src0_dd_i, src0_ddq_as_f32.get(), row_diff*ne00, stream);
        }
        if (src1->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp32");
            const to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(src1->type, dst);
            GGML_ASSERT(to_fp32_sycl != nullptr);
            src1_ddq_as_f32.alloc(src1_ncols*ne10);
            to_fp32_sycl(src1_ddf_i, src1_ddq_as_f32.get(), src1_ncols*ne10, stream);
        }
        const float * src0_ddf_i = src0->type == GGML_TYPE_F32 ? (const float *) src0_dd_i : src0_ddq_as_f32.get();
        const float * src1_ddf1_i = src1->type == GGML_TYPE_F32 ? (const float *) src1_ddf_i : src1_ddq_as_f32.get();

        {
#if GGML_SYCL_DNNL
            const int64_t gemm_flops = (int64_t)row_diff * src1_ncols * ne10;
            const bool use_mkl_direct = gemm_flops < 256 * 256 * 256;
            if (g_ggml_sycl_enable_dnn && !use_mkl_direct) {
                DnnlGemmWrapper::row_gemm(ctx, row_diff, src1_ncols, ne10, src0_ddf_i,
                                          DnnlGemmWrapper::to_dt<float>(), src1_ddf1_i, DnnlGemmWrapper::to_dt<float>(),
                                          dst_dd_i, DnnlGemmWrapper::to_dt<float>(), stream);
            }
            else
#endif
            {
                const float alpha = 1.0f;
                const float beta  = 0.0f;
                SYCL_CHECK(CHECK_TRY_ERROR(oneapi::mkl::blas::column_major::gemm(
                    *stream, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, row_diff,
                    src1_ncols, ne10, dpct::get_value(&alpha, *stream), src0_ddf_i, ne00, src1_ddf1_i, ne10,
                    dpct::get_value(&beta, *stream), dst_dd_i, ldc)));
            }
        }
    }
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(src1_padded_row_size);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

inline void ggml_sycl_op_sum(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const int64_t ne = ggml_nelements(dst->src[0]);

    sum_rows_f32_sycl(src0_dd, dst_dd, ne, 1, main_stream);
}

inline void ggml_sycl_op_sum_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const int64_t ncols = dst->src[0]->ne[0];
    const int64_t nrows = ggml_nrows(dst->src[0]);

    sum_rows_f32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);
}

inline void ggml_sycl_op_mean(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const int64_t ncols = dst->src[0]->ne[0];
    const int64_t nrows = ggml_nrows(dst->src[0]);

    sum_rows_f32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);

    main_stream->parallel_for(
        sycl::range<1>(nrows),
        [=](sycl::id<1> row) {
            dst_dd[row] /= ncols;
        }
    );
}


inline void ggml_sycl_op_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    int32_t *       dst_dd  = static_cast<int32_t *>(dst->data);


    const int64_t ncols = dst->src[0]->ne[0];
    const int64_t nrows = ggml_nrows(dst->src[0]);

    enum ggml_sort_order order = (enum ggml_sort_order) dst->op_params[0];

    argsort_f32_i32_sycl(src0_dd, (int *)dst_dd, ncols, nrows, order,
                         main_stream, ctx.device, ctx.pool());
}

// ARC-LAB: two-pass top-k for long rows (the MTP draft picks its tokens from the 248K-entry vocabulary: the one-group
// kernel above took ~550 us per call on the B580, one work-group on one Xe core plus a serial merge by one thread).
// Pass 1: up to 256 groups per row, each thread keeps a sorted top-k of a strided slice in registers, then the group
// merges its 256 lists pairwise in local memory (log2 steps of an O(k) merge). Pass 2: one group per row merges the
// per-group winners the same way. Ties keep the lower index.
namespace topk2 {
constexpr int WG = 256;

template <int K>
static inline void insert(float (&v)[K], int (&ix)[K], const int k, const float val, const int idx) {
    if (!(val > v[k - 1] || (val == v[k - 1] && idx < ix[k - 1] && ix[k - 1] >= 0))) {
        return;
    }
    int pos = k - 1;
    while (pos > 0 && (val > v[pos - 1] || (val == v[pos - 1] && idx < ix[pos - 1]))) {
        v[pos]  = v[pos - 1];
        ix[pos] = ix[pos - 1];
        pos--;
    }
    v[pos]  = val;
    ix[pos] = idx;
}

// merge the sorted lists of threads [0, WG) in sv/si (stride k) down to thread 0; every thread calls it
template <int K>
static inline void tree_merge(const sycl::nd_item<2> & it, float * sv, int * si, const int k, const int tid) {
    for (int s = WG / 2; s > 0; s >>= 1) {
        float mv[K];
        int   mi[K];
        if (tid < s) {
            const float * av = sv + tid * k;
            const int *   ai = si + tid * k;
            const float * bv = sv + (tid + s) * k;
            const int *   bi = si + (tid + s) * k;
            int a = 0, b = 0;
            for (int o = 0; o < k; ++o) {
                const bool take_a = av[a] > bv[b] || (av[a] == bv[b] && (bi[b] < 0 || (ai[a] >= 0 && ai[a] < bi[b])));
                if (take_a) {
                    mv[o] = av[a];
                    mi[o] = ai[a];
                    ++a;
                } else {
                    mv[o] = bv[b];
                    mi[o] = bi[b];
                    ++b;
                }
            }
        }
        it.barrier(sycl::access::fence_space::local_space);
        if (tid < s) {
            for (int o = 0; o < k; ++o) {
                sv[tid * k + o] = mv[o];
                si[tid * k + o] = mi[o];
            }
        }
        it.barrier(sycl::access::fence_space::local_space);
    }
}

template <int K>
static void run(const float * src, int32_t * dst, const int64_t ncols, const int64_t nrows, const int k,
                ggml_backend_sycl_context & ctx, dpct::queue_ptr q) {
    const int nblk = (int) std::min<int64_t>(256, (ncols + 2047) / 2048);
    ggml_sycl_pool_alloc<float> cv(ctx.pool(), (size_t) nrows * nblk * k);
    ggml_sycl_pool_alloc<int>   ci(ctx.pool(), (size_t) nrows * nblk * k);
    float * cvp = cv.get();
    int *   cip = ci.get();
    q->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sv(sycl::range<1>(WG * k), cgh);
        sycl::local_accessor<int, 1>   si(sycl::range<1>(WG * k), cgh);
        cgh.parallel_for(sycl::nd_range<2>({ (size_t) nrows, (size_t) nblk * WG }, { 1, WG }), [=](sycl::nd_item<2> it) {
            const int64_t row = it.get_group(0);
            const int     blk = it.get_group(1);
            const int     tid = it.get_local_id(1);
            const float * r   = src + row * ncols;
            float v[K];
            int   ix[K];
            for (int i = 0; i < k; ++i) {
                v[i]  = -FLT_MAX;
                ix[i] = -1;
            }
            for (int64_t c = (int64_t) blk * WG + tid; c < ncols; c += (int64_t) nblk * WG) {
                insert<K>(v, ix, k, r[c], (int) c);
            }
            float * svp = sv.template get_multi_ptr<sycl::access::decorated::no>().get();
            int *   sip = si.template get_multi_ptr<sycl::access::decorated::no>().get();
            for (int i = 0; i < k; ++i) {
                svp[tid * k + i] = v[i];
                sip[tid * k + i] = ix[i];
            }
            it.barrier(sycl::access::fence_space::local_space);
            tree_merge<K>(it, svp, sip, k, tid);
            if (tid < k) {
                cvp[(row * nblk + blk) * k + tid] = svp[tid];
                cip[(row * nblk + blk) * k + tid] = sip[tid];
            }
        });
    });
    q->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sv(sycl::range<1>(WG * k), cgh);
        sycl::local_accessor<int, 1>   si(sycl::range<1>(WG * k), cgh);
        cgh.parallel_for(sycl::nd_range<2>({ (size_t) nrows, (size_t) WG }, { 1, WG }), [=](sycl::nd_item<2> it) {
            const int64_t row = it.get_group(0);
            const int     tid = it.get_local_id(1);
            float v[K];
            int   ix[K];
            for (int i = 0; i < k; ++i) {
                v[i]  = -FLT_MAX;
                ix[i] = -1;
            }
            for (int c = tid; c < nblk * k; c += WG) {
                const int id = cip[row * nblk * k + c];
                if (id >= 0) {
                    insert<K>(v, ix, k, cvp[row * nblk * k + c], id);
                }
            }
            float * svp = sv.template get_multi_ptr<sycl::access::decorated::no>().get();
            int *   sip = si.template get_multi_ptr<sycl::access::decorated::no>().get();
            for (int i = 0; i < k; ++i) {
                svp[tid * k + i] = v[i];
                sip[tid * k + i] = ix[i];
            }
            it.barrier(sycl::access::fence_space::local_space);
            tree_merge<K>(it, svp, sip, k, tid);
            if (tid < k) {
                dst[row * k + tid] = sip[tid];
            }
        });
    });
}
}  // namespace topk2

static void ggml_sycl_op_top_k(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const float * src0_dd = static_cast<const float *>(src0->data);
    int32_t * dst_dd = static_cast<int32_t *>(dst->data);

    const int k = dst->ne[0];
    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    GGML_ASSERT(k > 0 && k <= 32);
    GGML_ASSERT(k <= ncols);

    static const bool old_topk = getenv("GGML_SYCL_TOPK_OLD") != nullptr;  // ARC-LAB A/B switch
    if (!old_topk && ncols >= 8192) {
        if (k <= 8) {
            topk2::run<8>(src0_dd, dst_dd, ncols, nrows, k, ctx, main_stream);
        } else if (k <= 16) {
            topk2::run<16>(src0_dd, dst_dd, ncols, nrows, k, ctx, main_stream);
        } else {
            topk2::run<32>(src0_dd, dst_dd, ncols, nrows, k, ctx, main_stream);
        }
        return;
    }
    top_k_f32_sycl(src0_dd, dst_dd, ncols, nrows, k, main_stream);
}

inline void ggml_sycl_op_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    int32_t *       dst_dd  = static_cast<int32_t *>(dst->data);

    const int64_t ncols = dst->src[0]->ne[0];
    const int64_t nrows = ggml_nrows(dst->src[0]);

    argmax_f32_i32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);
}

inline void ggml_sycl_op_diag_mask_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const int64_t ne00 = dst->src[0]->ne[0];
    const int64_t ne01 = dst->src[0]->ne[1];
    const int nrows0 = ggml_nrows(dst->src[0]);

    const int n_past = ((int32_t *) dst->op_params)[0];

    diag_mask_inf_f32_sycl(src0_dd, dst_dd, ne00, nrows0, ne01, n_past, main_stream);
}

static void tri_f32_sycl(
    const float * src,
    float * dst,
    const int64_t ne0,
    const int64_t ne1,
    const int64_t ne2,
    const int64_t ne3,
    const ggml_tri_type ttype,
    dpct::queue_ptr main_stream
) {
    const size_t total = (size_t) ne0 * (size_t) ne1 * (size_t) ne2 * (size_t) ne3;

    main_stream->parallel_for(sycl::range<1>(total), [=](sycl::id<1> tid) {
        const int64_t idx = (int64_t) tid[0];

        const int64_t i0 = idx % ne0;
        const int64_t t1 = idx / ne0;
        const int64_t i1 = t1 % ne1;

        bool keep = false;
        switch (ttype) {
            case GGML_TRI_TYPE_LOWER:      keep = (i0 <  i1); break;
            case GGML_TRI_TYPE_LOWER_DIAG: keep = (i0 <= i1); break;
            case GGML_TRI_TYPE_UPPER:      keep = (i0 >  i1); break;
            case GGML_TRI_TYPE_UPPER_DIAG: keep = (i0 >= i1); break;
            default: keep = false; break;
        }

        dst[idx] = keep ? src[idx] : 0.0f;
    });
}

static void ggml_sycl_op_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0);

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const float * src0_dd = static_cast<const float *>(src0->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);

    const int64_t ne0 = src0->ne[0];
    const int64_t ne1 = src0->ne[1];
    const int64_t ne2 = src0->ne[2];
    const int64_t ne3 = src0->ne[3];

    tri_f32_sycl(src0_dd, dst_dd, ne0, ne1, ne2, ne3, ttype, main_stream);
}


inline void ggml_sycl_op_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    float scale;
    float bias;
    memcpy(&scale, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&bias,  (float *) dst->op_params + 1, sizeof(float));

    scale_f32_sycl(src0_dd, dst_dd, scale, bias, ggml_nelements(dst->src[0]), main_stream);
    /*
    DPCT1010:87: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    SYCL_CHECK(0);
}

static void ggml_sycl_set_peer_access(const int n_tokens, int main_device) {
    static bool peer_access_enabled = false;

    const bool enable_peer_access = n_tokens <= GGML_SYCL_PEER_MAX_BATCH_SIZE;

    if (peer_access_enabled == enable_peer_access) {
        return;
    }

#ifdef NDEBUG
    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        SYCL_CHECK(ggml_sycl_set_device(i));
    }

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        SYCL_CHECK(ggml_sycl_set_device(i));

        for (int id_other = 0; id_other < ggml_sycl_info().device_count; ++id_other) {
            if (i == id_other) {
                continue;
            }
            if (i != main_device && id_other != main_device) {
                continue;
            }

            // int can_access_peer;
            // SYCL_CHECK(syclDeviceCanAccessPeer(&can_access_peer, id, id_other));
            // if (can_access_peer) {
            //     if (enable_peer_access) {
            //         SYCL_CHECK(syclDeviceEnablePeerAccess(id_other, 0));
            //     } else {
            //         SYCL_CHECK(syclDeviceDisablePeerAccess(id_other));
            //     }
            // }
        }
    }
#endif // NDEBUG

    peer_access_enabled = enable_peer_access;
}

template <template <int> typename quantize_f>
static void ggml_sycl_op_mul_mat(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                 const ggml_tensor *src1, ggml_tensor *dst,
                                 ggml_sycl_op_mul_mat_t op) try {

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne);

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne);
    const int64_t nrows1 = ggml_nrows(src1);

    GGML_ASSERT(ne03 == ne13);

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    const int nb2 = dst->nb[2];
    const int nb3 = dst->nb[3];

    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(dst->buffer));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src1->buffer));
    GGML_ASSERT(src1->type == GGML_TYPE_F32 || (src1->ne[2] == 1 && src1->ne[3] == 1));

    GGML_ASSERT(ne12 >= ne02 && ne12 % ne02 == 0);

    const int64_t i02_divisor = ne12 / ne02;

    const size_t src0_ts = ggml_type_size(src0->type);
    const size_t src0_bs = ggml_blck_size(src0->type);
    const size_t q8_1_ts = sizeof(block_q8_1);
    const size_t q8_1_bs = QK8_1;

    ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;
    ggml_tensor_extra_gpu * src1_extra = (ggml_tensor_extra_gpu *) src1->extra;

    const bool src0_is_contiguous = ggml_is_contiguous(src0);
    const bool src1_is_contiguous = ggml_is_contiguous(src1);

    int64_t src1_padded_col_size = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const bool split = ggml_backend_buffer_is_sycl_split(src0->buffer);
    GGML_ASSERT(!(split && ne02 > 1));
    GGML_ASSERT(!(split && ne03 > 1));
    GGML_ASSERT(!(split && ne02 < ne12));

    std::array<float, GGML_SYCL_MAX_DEVICES> tensor_split;
    if (split) {
        // TODO: check that src0->buffer->buft is a split buffer type, replace GGML_BACKEND_TYPE_GPU_SPLIT check
        // GGML_ASSERT(src0->buffer != nullptr && src0->buffer->buft == ...);
        ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *) src0->buffer->buft->context;
        tensor_split = buft_ctx->tensor_split;
    }

    struct dev_data {
        ggml_sycl_pool_alloc<char> src0_dd_alloc;
        ggml_sycl_pool_alloc<float> src1_ddf_alloc;
        ggml_sycl_pool_alloc<char> src1_ddq_alloc;
        ggml_sycl_pool_alloc<float> dst_dd_alloc;

        char *src0_dd = nullptr;
        float *src1_ddf = nullptr; // float
        char *src1_ddq = nullptr;  // q8_1
        float *dst_dd = nullptr;

        int64_t row_low;
        int64_t row_high;
    };

    dev_data dev[GGML_SYCL_MAX_DEVICES];

    int used_devices = 0;
    queue_ptr main_stream = ctx.stream();

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        // by default, use all rows
        dev[i].row_low  = 0;
        dev[i].row_high = ne01;

        // for multi GPU, get the row boundaries from tensor split
        // and round to mul_mat_q tile sizes
        if (split) {
            const int64_t rounding = get_row_rounding(src0->type, tensor_split);

            if (i != 0) {
                dev[i].row_low  = ne01*tensor_split[i];
                if (dev[i].row_low < ne01) {
                    dev[i].row_low -= dev[i].row_low % rounding;
                }
            }

            if (i != ggml_sycl_info().device_count - 1) {
                dev[i].row_high  = ne01*tensor_split[i + 1];
                if (dev[i].row_high < ne01) {
                    dev[i].row_high -= dev[i].row_high % rounding;
                }
            }
        }
    }

    constexpr bool quantize_enabled = !std::is_same_v<quantize_f<QK8_1 / WARP_SIZE>,
                                                      no_quantize_q8_1<QK8_1 / WARP_SIZE>>;
    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        if ((!split && i != ctx.device) || dev[i].row_low == dev[i].row_high) {
            continue;
        }

        used_devices++;

        const bool src1_on_device = i == ctx.device;
        const bool  dst_on_device = i == ctx.device;

        ggml_sycl_set_device(i);
        queue_ptr stream = ctx.stream(i, 0);

        if (src0_is_contiguous) {
            dev[i].src0_dd = (char *) src0->data;
        } else {
            dev[i].src0_dd = dev[i].src0_dd_alloc.alloc(ctx.pool(i), ggml_nbytes(src0));
        }

        if (src1_on_device && src1_is_contiguous) {
            dev[i].src1_ddf = (float *) src1->data;
        } else {
            dev[i].src1_ddf = dev[i].src1_ddf_alloc.alloc(ctx.pool(i), ggml_nelements(src1));
        }

        if constexpr(quantize_enabled) {
            dev[i].src1_ddq = dev[i].src1_ddq_alloc.alloc(ctx.pool(i), nrows1*src1_padded_col_size*q8_1_ts/q8_1_bs);

            if (src1_on_device && src1_is_contiguous) {
                scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                     /*num_src=*/2, " : converting src1 to Q8_1");
                try {
                    quantize_row_q8_1_sycl<quantize_f>(dev[i].src1_ddf, dev[i].src1_ddq, ne10, nrows1, src1_padded_col_size, stream);
                } catch (sycl::exception const &exc) {
                    std::cerr << "Quantize_row_q8_1_sycl error" << exc.what() << "Exception caught at file:" << __FILE__
                              << ", line:" << __LINE__ << std::endl;
                    std::exit(1);
                }
            }
        }

        if (dst_on_device) {
            dev[i].dst_dd = (float *) dst->data;
        } else {
            const size_t size_dst_ddf = split ? (dev[i].row_high - dev[i].row_low)*ne1 : ggml_nelements(dst);
            dev[i].dst_dd = dev[i].dst_dd_alloc.alloc(ctx.pool(i), size_dst_ddf);
        }
    }

    // if multiple devices are used they need to wait for the main device
    // here an event is recorded that signals that the main device has finished calculating the input data
    if (split && used_devices > 1) {
        ggml_sycl_set_device(ctx.device);
        SYCL_CHECK(CHECK_TRY_ERROR(
            *src0_extra->events[ctx.device][0] =
                ctx.stream()->ext_oneapi_submit_barrier()));
    }

    const int64_t src1_col_stride = split && used_devices > 1 ? MUL_MAT_SRC1_COL_STRIDE : ne11;
    for (int64_t src1_col_0 = 0; src1_col_0 < ne11; src1_col_0 += src1_col_stride) {
        const int64_t is = split ? (src1_col_0/src1_col_stride) % GGML_SYCL_MAX_STREAMS : 0;
        const int64_t src1_ncols = src1_col_0 + src1_col_stride > ne11 ? ne11 - src1_col_0 : src1_col_stride;
        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            if ((!split && i != ctx.device) || dev[i].row_low == dev[i].row_high) {
                continue;
            }

            const bool src1_on_device = i == ctx.device;
            const bool  dst_on_device = i == ctx.device;
            const int64_t row_diff = dev[i].row_high - dev[i].row_low;

            ggml_sycl_set_device(i);
            queue_ptr stream = ctx.stream(i, is);

            // wait for main GPU data if necessary
            if (split && (i != ctx.device || is != 0)) {
                SYCL_CHECK(CHECK_TRY_ERROR(stream->ext_oneapi_submit_barrier(
                    {*src0_extra->events[ctx.device][0]})));
            }

            for (int64_t i0 = 0; i0 < ne13*ne12; ++i0) {
                const int64_t i03 = i0 / ne12;
                const int64_t i02 = i0 % ne12;

                const size_t src1_ddq_i_offset = (i0*ne11 + src1_col_0) * src1_padded_col_size*q8_1_ts/q8_1_bs;

                // for split tensors the data begins at i0 == i0_offset_low
                char  *  src0_dd_i =  dev[i].src0_dd + (i0/i02_divisor) * (ne01*ne00*src0_ts)/src0_bs;
                float * src1_ddf_i = dev[i].src1_ddf + (i0*ne11 + src1_col_0) * ne10;
                char  * src1_ddq_i = dev[i].src1_ddq +  src1_ddq_i_offset;
                float *   dst_dd_i =   dev[i].dst_dd + (i0*ne1  + src1_col_0) * (dst_on_device ? ne0 : row_diff);

                // the main device memory buffer can be on VRAM scratch, with space for all partial results
                // in that case an offset on dst_ddf_i is needed
                if (i == ctx.device) {
                    dst_dd_i += dev[i].row_low; // offset is 0 if no tensor split
                }

                // copy src0, src1 to device if necessary
                if (src1_is_contiguous) {
                    if (i != ctx.device) {
                        if constexpr (quantize_enabled) {
                            char * src1_ddq_i_source = dev[ctx.device].src1_ddq + src1_ddq_i_offset;
                            SYCL_CHECK(
                                CHECK_TRY_ERROR(stream
                                                    ->memcpy(src1_ddq_i, src1_ddq_i_source,
                                                             src1_ncols * src1_padded_col_size * q8_1_ts / q8_1_bs)
                                                    .wait()));
                        } else {
                            float * src1_ddf_i_source = (float *) src1_extra->data_device[ctx.device];
                            src1_ddf_i_source += (i0 * ne11 + src1_col_0) * ne10;

                            SYCL_CHECK(
                                CHECK_TRY_ERROR(dev2dev_memcpy(i, *stream, ctx.device, *main_stream, src1_ddf_i, src1_ddf_i_source,
                                                               src1_ncols * ne10 * sizeof(float))));
                        }
                    }
                } else {
                    if (src1_on_device) {
                        SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src1_ddf_i, src1, i03, i02, src1_col_0,
                                                           src1_col_0 + src1_ncols, stream));
                    } else {
                        GGML_ABORT("src1 is non-contiguous and not on device");
                    }

                    if constexpr (quantize_enabled) {
                        scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                             /*num_src=*/2, " : converting src1 to Q8_1");
                        try {
                            quantize_row_q8_1_sycl<quantize_q8_1>(src1_ddf_i, src1_ddq_i, ne10, src1_ncols,
                                                                  src1_padded_col_size, stream);
                        } catch (const sycl::exception & exc) {
                            std::cerr << "Quantize_row_q8_1_sycl error" << exc.what()
                                      << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
                            std::exit(1);
                        }
                    }
                }

                if (src1_col_0 == 0 && !src0_is_contiguous && i02 % i02_divisor == 0) {
                    SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src0_dd_i, src0, i03, i02/i02_divisor, dev[i].row_low, dev[i].row_high, stream));
                }
                if (src1->type == GGML_TYPE_F16) {
                    src1_padded_col_size = (i0 * ne11 + src1_col_0) * ne10;
                }
                // do the computation
                SYCL_CHECK(CHECK_TRY_ERROR(op(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                    dev[i].row_low, dev[i].row_high, src1_ncols, src1_padded_col_size, stream)));

                // copy dst to host or other device if necessary
                if (!dst_on_device) {
                    void * dst_off_device = dst->data;
                    if (split) {
                        // src0 = weight matrix is saved as a transposed matrix for better memory layout.
                        // dst is NOT transposed.
                        // The outputs of matrix matrix multiplications can therefore NOT simply be concatenated for >1 GPU.
                        // Instead they need to be copied to the correct slice in ne0 = dst row index.
                        // If dst is a vector with ne0 == 1 then you don't have to do this but it still produces correct results.
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*sizeof(float));
                        dhf_dst_i += src1_col_0*ne0 + dev[i].row_low;

                        SYCL_CHECK(CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
                            dhf_dst_i, ne0 * sizeof(float), dst_dd_i,
                            row_diff * sizeof(float), row_diff * sizeof(float),
                            src1_ncols, dpct::device_to_device, *stream)));
                    } else {
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*sizeof(float));
                        dhf_dst_i += src1_col_0*ne0;
                        SYCL_CHECK(CHECK_TRY_ERROR(
                            stream->memcpy(dhf_dst_i, dst_dd_i,
                                           src1_ncols * ne0 * sizeof(float)).wait()));
                    }
                }

                // add event for the main device to wait on until other device is done
                if (split && (i != ctx.device || is != 0)) {
                    SYCL_CHECK(CHECK_TRY_ERROR(
                        *src0_extra->events[i][is] =
                            stream->ext_oneapi_submit_barrier()));
                }
            }
        }
    }

    // main device waits for all other devices to be finished
    if (split && ggml_sycl_info().device_count > 1) {
        int64_t is_max = (ne11 + MUL_MAT_SRC1_COL_STRIDE - 1) / MUL_MAT_SRC1_COL_STRIDE;
        is_max = is_max <= GGML_SYCL_MAX_STREAMS ? is_max : GGML_SYCL_MAX_STREAMS;

        ggml_sycl_set_device(ctx.device);
        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            if (dev[i].row_low == dev[i].row_high) {
                continue;
            }
            for (int64_t is = 0; is < is_max; ++is) {
                SYCL_CHECK(CHECK_TRY_ERROR(
                    ctx.stream()->ext_oneapi_submit_barrier(
                        {*src0_extra->events[i][is]})));
            }
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_sycl_repeat_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_repeat_back(ctx, dst);
}

static void ggml_sycl_get_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_get_rows(ctx, dst);
}

static void ggml_sycl_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_norm(ctx, dst);
}

static void ggml_sycl_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_rms_norm(ctx, dst);
}

static void ggml_sycl_rms_norm_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_rms_norm_back(ctx, dst);
}

static void ggml_sycl_l2_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_l2_norm(ctx, dst);
}

static void ggml_sycl_group_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_group_norm(ctx, dst);
}

static void ggml_sycl_mul_mat_vec_p021(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                       const ggml_tensor *src1,
                                       ggml_tensor *dst) try {
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer));
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]); // 0213 permutation
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]); // 0213 permutation
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    const int64_t ne12 = src1->ne[2];

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    void  * src0_ddq = src0->data;
    float * src1_ddf = (float *) src1->data;
    float * dst_ddf  = (float *) dst->data;

    ggml_mul_mat_p021_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, ne02, ne12, main_stream);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_sycl_mul_mat_vec_nc(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                     const ggml_tensor *src1,
                                     ggml_tensor *dst) try {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->ne[1] == 1);
    GGML_ASSERT(src1->ne[3] == 1);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];

    const int64_t ne12 = src1->ne[2];
    const int64_t nb11 = src1->nb[1];

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    void  * src0_ddq = src0->data;
    float * src1_ddf = (float *) src1->data;
    float * dst_ddf  = (float *) dst->data;

    const int64_t row_stride_x = nb01 / sizeof(sycl::half);
    const int64_t channel_stride_x = nb02 / sizeof(sycl::half);
    const int64_t channel_stride_y = nb11 / sizeof(float);

    ggml_mul_mat_vec_nc_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, row_stride_x, ne02, ne12, channel_stride_x,channel_stride_y, main_stream);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void k_compute_batched_ptrs(const sycl::half * src0_as_f16, const sycl::half * src1_as_f16, void * dst,
                                   const void ** ptrs_src, void ** ptrs_dst, int64_t ne12, int64_t ne13, int64_t ne23,
                                   size_t nb02, size_t nb03, size_t nb12, size_t nb13, size_t nbd2, size_t nbd3,
                                   int64_t r2, int64_t r3, const sycl::nd_item<3> & item_ct1) {
    const int64_t i13 = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    const int64_t i12 = item_ct1.get_group(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    const int64_t i03 = i13 / r3;
    const int64_t i02 = i12 / r2;

    const uint8_t * src0_bytes = reinterpret_cast<const uint8_t *>(src0_as_f16);
    const uint8_t * src1_bytes = reinterpret_cast<const uint8_t *>(src1_as_f16);
    uint8_t *       dst_bytes  = static_cast<uint8_t *>(dst);

    ptrs_src[0 * ne23 + i12 + i13 * ne12] = src0_bytes + i02 * nb02 + i03 * nb03;
    ptrs_src[1 * ne23 + i12 + i13 * ne12] = src1_bytes + i12 * nb12 + i13 * nb13;
    ptrs_dst[0 * ne23 + i12 + i13 * ne12] = dst_bytes + i12 * nbd2 + i13 * nbd3;
}

static void ggml_sycl_mul_mat_batched_sycl(ggml_backend_sycl_context & ctx, const ggml_tensor * src0,
                                           const ggml_tensor * src1, ggml_tensor * dst) try {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    // TODO: see https://github.com/ggml-org/llama.cpp/pull/13155
    // Batched mul_mat requires a rewrite to support both oneDNN and non-contiguous dst
    GGML_ASSERT(ggml_is_contiguous(dst));

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr queue = ctx.stream();

    dpct::has_capability_or_fail(queue->get_device(), { sycl::aspect::fp16 });

    const sycl::half * src0_f16 = static_cast<const sycl::half *>(src0->data);
    float *            dst_ddf  = static_cast<float *>(dst->data);

    const sycl::half * src1_f16       = static_cast<const sycl::half *>(src1->data);
#if GGML_SYCL_DNNL
    const size_t       type_size_src0 = ggml_type_size(src0->type);
#endif
    const size_t       type_size_src1 = ggml_type_size(src1->type);

    bool is_src0_cont_2 = ggml_is_contiguous_2(src0);
    bool is_src1_cont_2 = ggml_is_contiguous_2(src1);

    // SRC1 strides
    int64_t                          s11 = nb11 / type_size_src1;
    int64_t                          s12 = nb12 / type_size_src1;
    int64_t                          s13 = nb13 / type_size_src1;
    ggml_sycl_pool_alloc<sycl::half> src1_f16_alloc(ctx.pool());

    // convert src1 to fp16
    if (src1->type != GGML_TYPE_F16) {
        scope_op_debug_print    scope_dbg_print(__func__, "/to_fp16_nc_sycl", dst, /*num_src=*/2,
                                                " : converting src1 to fp16");

#if GGML_SYCL_DNNL
        if (g_ggml_sycl_enable_dnn) {
        // iterate tensor dims and find the slowest moving dim and stride
        int last_dim=0;
        int last_str=0;
        size_t largest_str=0;
        for(int i = 0; i< 4; i++){
            // last stride is always the largest
            if(src1->nb[i] == largest_str){
                if(src1->ne[last_dim] == 1){
                    last_str = i;
                    last_dim = i;
                }
            }
            if(src1->nb[i] > largest_str){
                largest_str = src1->nb[i];
                last_str = i;
                last_dim = i;
            }

        }
        // oneDNN handles strided data and does not need overhead of ggml_get_to_fp16_nc_sycl
        const int64_t ne_src1 = src1->nb[last_str] * src1->ne[last_dim] / type_size_src1;
        src1_f16_alloc.alloc(ne_src1);
        const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type, dst);
        GGML_ASSERT(to_fp16_sycl != nullptr);
        to_fp16_sycl(src1_f16, src1_f16_alloc.get(), ne_src1, queue);
        } else
#endif
        {
        const int64_t ne_src1 = ggml_nelements(src1);
        src1_f16_alloc.alloc(ne_src1);
        const to_fp16_nc_sycl_t to_fp16_nc_sycl = ggml_get_to_fp16_nc_sycl(src1->type);
        GGML_ASSERT(to_fp16_nc_sycl != nullptr);
        to_fp16_nc_sycl(src1_f16, src1_f16_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, queue);
        }

        src1_f16 = src1_f16_alloc.get();
        s11      = ne10;
        s12      = ne11 * s11;
        s13      = ne12 * s12;

        is_src1_cont_2 = true;
    }

    ggml_sycl_pool_alloc<sycl::half> dst_f16(ctx.pool());

    dpct::library_data_t mkl_compute_type = dpct::library_data_t::real_float;
    dpct::library_data_t mkl_data_type    = dpct::library_data_t::real_float;

    // dst strides
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    const float alpha_f32 = 1.0f;
    const float beta_f32  = 0.0f;

    const void * alpha = &alpha_f32;
    const void * beta  = &beta_f32;

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);
    GGML_ASSERT(ne01 == static_cast<int64_t>(nb1/nb0));
    GGML_ASSERT(ne10 == ne00);

    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

#if GGML_SYCL_DNNL
    if (g_ggml_sycl_enable_dnn) {
            int64_t str_a0 = nb00 / type_size_src0;
            int64_t str_a1 = nb01 / type_size_src0;
            int64_t str_a2 = nb02 / type_size_src0;

            int64_t str_b0 = nb10 / type_size_src1;
            int64_t str_b1 = nb11 / type_size_src1;
            int64_t str_b2 = nb12 / type_size_src1;

            auto launch_gemm_for_batches = [&ctx, queue](const sycl::half *src0,
                                                const sycl::half *src1, float *dst,
                                                int64_t a0, int64_t a1, int64_t batcha,
                                                int64_t /*b0*/, int64_t b1, int64_t batchb,
                                                int64_t sa0, int64_t sa1, int64_t sa2,
                                                int64_t sb0, int64_t sb1, int64_t sb2,
                                                int64_t sd2) {
                bool supported_broadcast = batchb == batcha ? true
                        : batchb == 1 || batcha == 1        ? true
                                                            : false;
                if (supported_broadcast) {
                    DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0,
                            DnnlGemmWrapper::to_dt<sycl::half>(), sa0, sa1, sa2, src1,
                            DnnlGemmWrapper::to_dt<sycl::half>(), sb0, sb1, sb2, dst,
                            DnnlGemmWrapper::to_dt<float>(), queue, batcha, batchb);
                } else {
                    // iterate over batches from smaller set of matrices (matrix 0)
                    int64_t batches0 = batcha;
                    int64_t batches1 = batchb;

                    if (batches0 > batches1) {
                        int64_t num_mul_mats = batches1;
                        int64_t sub_batch = batches0 / num_mul_mats;
                        // src0 is batched and bigger, shift and multiply with src1
                        for (int64_t i0 = 0; i0 < num_mul_mats; i0++) {
                            const sycl::half *src0_shifted = src0 + (sa2 * i0 * sub_batch);
                            const sycl::half *src1_shifted = src1 + (sb2 * i0);
                            float *dst_shifted = dst + (sd2 * i0 * sub_batch);
                            DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted,
                                    DnnlGemmWrapper::to_dt<sycl::half>(), sa0, sa1, sa2,
                                    src1_shifted, DnnlGemmWrapper::to_dt<sycl::half>(), sb0,
                                    sb1, sb2, dst_shifted, DnnlGemmWrapper::to_dt<float>(),
                                    queue, sub_batch, 1);
                        }
                    } else {
                        int64_t num_mul_mats = batches0;
                        int64_t sub_batch = batches1 / num_mul_mats;
                        // src1 is batched and bigger, shift and multiply with src0
                        for (int64_t i1 = 0; i1 < num_mul_mats; i1++) {
                            const sycl::half *src0_shifted = src0 + (sa2 * i1);
                            const sycl::half *src1_shifted = src1 + (sb2 * i1 * sub_batch);
                            float *dst_shifted = dst + (sd2 * i1 * sub_batch);
                            DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted,
                                    DnnlGemmWrapper::to_dt<sycl::half>(), sa0, sa1, sa2,
                                    src1_shifted, DnnlGemmWrapper::to_dt<sycl::half>(), sb0,
                                    sb1, sb2, dst_shifted, DnnlGemmWrapper::to_dt<float>(),
                                    queue, 1, sub_batch);
                        }
                    }
                }
            };

            const bool cont_batches_dim2_a = nb02 * ne02 == nb03;
            const bool cont_batches_dim2_b = nb12 * ne12 == nb13;
            const bool cont_batches_dim3_a = ne02 == 1 && nb02 * ne01 == nb03;
            const bool cont_batches_dim3_b = ne12 == 1 && nb12 * ne11 == nb13;
            if (cont_batches_dim2_a && cont_batches_dim2_b) {
                // A batch is considered contiguous if the dimension 2 is not strided
                int64_t batches0 = ne02 * ne03;
                int64_t batches1 = ne12 * ne13;
                launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0,
                        ne10, ne11, batches1, str_a0, str_a1, str_a2, str_b0, str_b1,
                        str_b2, nb2 / sizeof(float));
            } else if (cont_batches_dim3_a && cont_batches_dim3_b) {
                // This case is similar to the one above with the difference that only the batch in dimension 3 is used and the dimension 2 is of size 1.
                int64_t batches0 = ne02 * ne03;
                int64_t batches1 = ne12 * ne13;
                int64_t str_a3 = nb03 / type_size_src0;
                int64_t str_b3 = nb13 / type_size_src1;
                launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0,
                        ne10, ne11, batches1, str_a0, str_a1, str_a3, str_b0, str_b1,
                        str_b3, nb2 / sizeof(float));
            } else {
                for (int64_t b_a = 0; b_a < ne03; b_a++) {
                    const sycl::half *src0_f16_shifted
                            = src0_f16 + (nb03 * b_a / type_size_src0);
                    const sycl::half *src1_f16_shifted
                            = src1_f16 + (nb13 * b_a / type_size_src1);
                    float *dst_shifted = dst_ddf + (nb3 * b_a / sizeof(float));
                    int64_t batches0 = ne02;
                    int64_t batches1 = ne12;
                    launch_gemm_for_batches(src0_f16_shifted, src1_f16_shifted, dst_shifted,
                            ne00, ne01, batches0, ne10, ne11, batches1, str_a0, str_a1,
                            str_a2, str_b0, str_b1, str_b2, nb2 / sizeof(float));
                }
            }

    }
    else
#endif
    {
        if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
            // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
            const int64_t sma = ne02 == 1 ? nb03/nb00 : nb02/nb00;
            const int64_t smb = ne12 == 1 ? s13       : s12;

            // there is no broadcast and src0, src1 are contiguous across dims 2, 3
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(*queue, oneapi::mkl::transpose::trans,
                                                        oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
                                                        src0_f16, dpct::library_data_t::real_half, nb01 / nb00, sma,
                                                        src1_f16, dpct::library_data_t::real_half, s11, smb, beta, dst_ddf,
                                                        mkl_data_type, ne0, ne1 * ne0, ne12 * ne13, mkl_compute_type)));
        } else {
            const int ne23 = ne12 * ne13;

            ggml_sycl_pool_alloc<const void *>         ptrs_src(ctx.pool(), 2 * ne23);
            ggml_sycl_pool_alloc<void *>               ptrs_dst(ctx.pool(), 1 * ne23);
            ggml_sycl_pool_alloc<matrix_info_t<float>> matrix_info(ctx.host_pool(), 1);

            sycl::range<3> block_dims(1, ne12, ne13);
            queue->submit([&](sycl::handler & cgh) {
                const void ** ptrs_src_get = ptrs_src.get();
                void **       ptrs_dst_get = ptrs_dst.get();
                size_t        nb12_scaled  = src1->type == GGML_TYPE_F16 ? nb12 : s12 * sizeof(sycl::half);
                size_t        nb13_scaled  = src1->type == GGML_TYPE_F16 ? nb13 : s13 * sizeof(sycl::half);
                cgh.parallel_for(sycl::nd_range<3>(block_dims, block_dims), [=](sycl::nd_item<3> item_ct1) {
                    k_compute_batched_ptrs(src0_f16, src1_f16, dst_ddf, ptrs_src_get, ptrs_dst_get, ne12, ne13, ne23, nb02,
                                           nb03, nb12_scaled, nb13_scaled, nbd2, nbd3, r2, r3, item_ct1);
                });
            });

            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(
                *queue, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
                (const void **) (ptrs_src.get() + 0 * ne23), dpct::library_data_t::real_half, nb01 / nb00,
                (const void **) (ptrs_src.get() + 1 * ne23), dpct::library_data_t::real_half, s11, beta,
                (void **) (ptrs_dst.get() + 0 * ne23), mkl_data_type, ne0, ne23, mkl_compute_type, matrix_info.get())));
        }
    }
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

enum class mul_mat_algo {
    DMMV         = 0,
    MMVQ         = 1,
    MUL_MAT_SYCL = 2,
};

inline bool ggml_sycl_supports_mmq(enum ggml_type type) {
    // TODO: accuracy issues in MMQ
    GGML_UNUSED(type);
    return false;
}

// The PQ2_0/PTQ1_0 XMX path feeds 2-bit weights to ESIMD DPAS at execution size 16 through 2D block loads, which
// every XMX device with 16-wide DPAS has (Xe-HPC, Xe2 and later). 8-wide XMX (Xe-HPG, Arrow Lake-H) keeps the
// existing paths.
static bool ggml_sycl_device_has_dpas16(int device) {
    return ggml_sycl_info().devices[device].dpas_exec_size == 16;
}

inline bool ggml_sycl_supports_reorder_mul_mat_sycl(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
            return true;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return !g_ggml_sycl_prioritize_dmmv;
        default:
            return false;
    }
}

inline bool ggml_sycl_supports_reorder_dmmv(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

inline bool ggml_sycl_supports_reorder_mmvq(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_PQ2_0:
        case GGML_TYPE_PTQ1_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

static bool ggml_sycl_supports_reorder_esimd(enum ggml_type type) {
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
    switch (type) {
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
#else
    GGML_UNUSED(type);
    return false;
#endif
}

static bool ggml_sycl_supports_dmmv(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            return true;
        default:
            return false;
    }
}

static bool ggml_sycl_supports_mmvq(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_PTQ1_0:
        case GGML_TYPE_PQ2_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            return true;
        default:
            return false;
    }
}

// Helper functions to unify device memory allocation for both async and sync paths
static inline void * sycl_ext_malloc_device(dpct::queue_ptr stream, size_t size) {
    bool use_async = g_ggml_sycl_use_async_mem_op;
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
    if (use_async) {
        return syclex::async_malloc(*stream, sycl::usm::alloc::device, size);
    }
#else
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#endif
    return ggml_sycl_malloc_device(size, *stream);
}

static inline void sycl_ext_free(dpct::queue_ptr stream, void * ptr) {
    bool use_async = g_ggml_sycl_use_async_mem_op;
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
    if (use_async) {
        syclex::async_free(*stream, ptr);
        return;
    }
#else
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#endif
    ggml_sycl_free_device(ptr, *stream);
}

// RAII wrapper for temporary reorder buffers with optional host memory fallback.
// When device allocation fails and GGML_SYCL_HOST_MEM_FALLBACK is enabled,
// falls back to host memory so the reorder kernel can still run (over PCIe).
// Device access to host memory requires Linux kernel 6.8+ (Ubuntu 26.04+).
struct sycl_reorder_temp_buffer {
    void *          ptr  = nullptr;
    dpct::queue_ptr stream;

    sycl_reorder_temp_buffer(dpct::queue_ptr stream, size_t size) : stream(stream) {
        ptr = sycl_ext_malloc_device(stream, size);
#ifdef GGML_SYCL_HOST_MEM_FALLBACK
        if (!ptr) {
            ptr = sycl::malloc_host(size, *stream);
            if (ptr) {
                host_fallback = true;
                GGML_LOG_WARN("%s: device alloc of %zu bytes failed, using host memory fallback\n", __func__, size);
            }
        }
#endif
    }

    ~sycl_reorder_temp_buffer() {
        if (!ptr) {
            return;
        }
        if (host_fallback) {
            sycl::free(ptr, *stream);
        } else {
            sycl_ext_free(stream, ptr);
        }
    }

    explicit operator bool() const { return ptr != nullptr; }

    sycl_reorder_temp_buffer(const sycl_reorder_temp_buffer &)            = delete;
    sycl_reorder_temp_buffer & operator=(const sycl_reorder_temp_buffer &) = delete;

private:
    bool host_fallback = false;
};

static bool reorder_qw_q4_0(uint8_t * data_device, const int ncols, const int nrows, size_t size, size_t offset,
                            dpct::queue_ptr stream) {
    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    GGML_ASSERT((size % sizeof(block_q4_0) == 0));
    GGML_ASSERT((offset % sizeof(block_q4_0) == 0));
    int offset_blks = offset / sizeof(block_q4_0);
    auto qs_ptr      = data_device + offset_blks * QK4_0 / 2;
    auto d_ptr = (sycl::half*)(qs_ptr + ncols * nrows / 2) + offset_blks;

    auto reorder_event = stream->parallel_for(
        size / sizeof(block_q4_0),
            [=](auto i) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            const block_q4_0* x = (const block_q4_0*)tmp_buf;
            const int ib = i;

            for (int j = 0; j < QK4_0/2; j ++)
            {
                *(qs_ptr + ib * QK4_0 / 2 + j) = x[ib].qs[j];
            }
            *(d_ptr + ib) = x[ib].d;
        });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

static bool reorder_qw_pq2_0(uint8_t * data_device, const int ncols, const int nrows, size_t size, size_t offset,
                             dpct::queue_ptr stream) {
    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    GGML_ASSERT((size % sizeof(block_pq2_0) == 0));
    GGML_ASSERT((offset % sizeof(block_pq2_0) == 0));
    int offset_blks = offset / sizeof(block_pq2_0);
    auto qs_ptr     = data_device + offset_blks * QK_PQ2_0 / 4;
    auto d_ptr      = (sycl::half *) (qs_ptr + ncols * nrows / 4) + offset_blks;

    auto reorder_event = stream->parallel_for(
        size / sizeof(block_pq2_0),
            [=](auto i) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            const block_pq2_0 * x = (const block_pq2_0 *) tmp_buf;
            const int ib = i;

            for (int j = 0; j < QK_PQ2_0 / 4; j++) {
                *(qs_ptr + ib * QK_PQ2_0 / 4 + j) = x[ib].qs[j];
            }
            *(d_ptr + ib) = x[ib].d;
        });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

static bool reorder_qw_ptq1_0(uint8_t * data_device, size_t size, dpct::queue_ptr stream) {
    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    GGML_ASSERT((size % sizeof(block_ptq1_0) == 0));
    const int64_t nb     = size / sizeof(block_ptq1_0);
    auto          qs_ptr = data_device;
    auto          qh_ptr = data_device + nb * 24;
    auto          d_ptr  = (sycl::half *) (data_device + nb * 26);

    auto reorder_event = stream->parallel_for(nb, [=](auto i) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
        const block_ptq1_0 * x  = (const block_ptq1_0 *) tmp_buf;
        const int64_t        ib = i;
        for (int j = 0; j < 24; j++) {
            qs_ptr[ib * 24 + j] = x[ib].qs[j];
        }
        qh_ptr[ib * 2 + 0] = x[ib].qh[0];
        qh_ptr[ib * 2 + 1] = x[ib].qh[1];
        d_ptr[ib]          = x[ib].d;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

static bool reorder_qw_q8_0(uint8_t * data_device, const int ncols, const int nrows, size_t size, size_t offset,
                            dpct::queue_ptr stream) {
    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    GGML_ASSERT((size % sizeof(block_q8_0) == 0));
    GGML_ASSERT((offset % sizeof(block_q8_0) == 0));
    int offset_blks = offset / sizeof(block_q8_0);
    auto qs_ptr = data_device + offset_blks * QK8_0;
    auto d_ptr = (sycl::half*)(qs_ptr + ncols * nrows) + offset_blks;

    auto reorder_event = stream->parallel_for(
        size / sizeof(block_q8_0),
            [=](auto i) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            const block_q8_0* x = (const block_q8_0*)tmp_buf;
            const int ib = i;

            for (int j = 0; j < QK8_0; j++)
            {
                *((int8_t*)qs_ptr + ib * QK8_0 + j) = x[ib].qs[j];
            }
            *(d_ptr + ib) = x[ib].d;
        });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

static bool reorder_qw_q4_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % sizeof(block_q4_K) == 0);
    GGML_ASSERT(offset % sizeof(block_q4_K) == 0);

    const int nblocks = size / sizeof(block_q4_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto * qs_ptr     = data_device;
    auto * scales_ptr = qs_ptr + QK_K / 2 * nblocks;
    auto * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        const block_q4_K * x  = (const block_q4_K *) tmp_buf;
        const int          ib = i;

        for (int j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }

        for (int j = 0; j < K_SCALE_SIZE; ++j) {
            scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

// Reorder each expert slice into a self-contained SoA layout.
static bool reorder_qw_q4_k_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % sizeof(block_q4_K) == 0);
    const int    blocks_per_expert = (int) (expert_bytes / sizeof(block_q4_K));
    const size_t total_bytes       = expert_bytes * (size_t) n_expert;

    sycl_reorder_temp_buffer tmp(stream, total_bytes);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, total_bytes);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, total_bytes)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    const int total_blocks = blocks_per_expert * (int) n_expert;
    auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
        const int          gb   = gb_;
        const int          e    = gb / blocks_per_expert;
        const int          ib   = gb % blocks_per_expert;
        const block_q4_K * x    = (const block_q4_K *) (tmp_buf + (size_t) e * expert_bytes);
        uint8_t *          base = data_device + (size_t) e * expert_bytes;

        auto * qs_ptr     = base;
        auto * scales_ptr = qs_ptr + QK_K / 2 * blocks_per_expert;
        auto * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * blocks_per_expert);

        for (int j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }
        for (int j = 0; j < K_SCALE_SIZE; ++j) {
            scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
        }
        dm_ptr[ib] = x[ib].dm;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

// Reorder each Q5_K expert slice into [qs][qh][scales][dm].
static bool reorder_qw_q5_k_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % sizeof(block_q5_K) == 0);
    const int    blocks_per_expert = (int) (expert_bytes / sizeof(block_q5_K));
    const size_t total_bytes       = expert_bytes * (size_t) n_expert;

    sycl_reorder_temp_buffer tmp(stream, total_bytes);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, total_bytes);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, total_bytes)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    const int total_blocks = blocks_per_expert * (int) n_expert;
    auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
        const int          gb   = gb_;
        const int          e    = gb / blocks_per_expert;
        const int          ib   = gb % blocks_per_expert;
        const block_q5_K * x    = (const block_q5_K *) (tmp_buf + (size_t) e * expert_bytes);
        uint8_t *          base = data_device + (size_t) e * expert_bytes;

        auto * qs_ptr     = base;
        auto * qh_ptr     = qs_ptr + (QK_K / 2) * blocks_per_expert;
        auto * scales_ptr = qh_ptr + (QK_K / 8) * blocks_per_expert;
        auto * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * blocks_per_expert);

        for (int j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }
        for (int j = 0; j < QK_K / 8; ++j) {
            qh_ptr[ib * (QK_K / 8) + j] = x[ib].qh[j];
        }
        for (int j = 0; j < K_SCALE_SIZE; ++j) {
            scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
        }
        dm_ptr[ib] = x[ib].dm;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

// Reorder each Q6_K expert slice into [ql][qh][scales][d].
static bool reorder_qw_q6_k_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % sizeof(block_q6_K) == 0);
    const int    blocks_per_expert = (int) (expert_bytes / sizeof(block_q6_K));
    const size_t total_bytes       = expert_bytes * (size_t) n_expert;

    sycl_reorder_temp_buffer tmp(stream, total_bytes);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, total_bytes);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, total_bytes)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    const int total_blocks = blocks_per_expert * (int) n_expert;
    auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
        const int          gb   = gb_;
        const int          e    = gb / blocks_per_expert;
        const int          ib   = gb % blocks_per_expert;
        const block_q6_K * x    = (const block_q6_K *) (tmp_buf + (size_t) e * expert_bytes);
        uint8_t *          base = data_device + (size_t) e * expert_bytes;

        auto * ql_ptr     = base;
        auto * qh_ptr     = ql_ptr + (QK_K / 2) * blocks_per_expert;
        auto * scales_ptr = qh_ptr + (QK_K / 4) * blocks_per_expert;
        auto * d_ptr      = (sycl::half *) (scales_ptr + (QK_K / 16) * blocks_per_expert);

        for (int j = 0; j < QK_K / 2; ++j) {
            ql_ptr[ib * (QK_K / 2) + j] = x[ib].ql[j];
        }
        for (int j = 0; j < QK_K / 4; ++j) {
            qh_ptr[ib * (QK_K / 4) + j] = x[ib].qh[j];
        }
        for (int j = 0; j < QK_K / 16; ++j) {
            scales_ptr[ib * (QK_K / 16) + j] = x[ib].scales[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

static bool reorder_qw_q2_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % sizeof(block_q2_K) == 0);
    GGML_ASSERT(offset % sizeof(block_q2_K) == 0);

    const int nblocks = size / sizeof(block_q2_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto *        qs_ptr     = data_device;
    auto *        scales_ptr = qs_ptr + (QK_K / 4) * nblocks;
    sycl::half2 * dm_ptr     = (sycl::half2 *) (scales_ptr + (QK_K / 16) * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        const block_q2_K * x  = (const block_q2_K *) tmp_buf;
        const int          ib = i;

        for (int j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }

        for (int j = 0; j < QK_K / 16; ++j) {
            scales_ptr[ib * (QK_K / 16) + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

static bool reorder_qw_q3_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % sizeof(block_q3_K) == 0);
    GGML_ASSERT(offset % sizeof(block_q3_K) == 0);

    const int nblocks = size / sizeof(block_q3_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto *       qs_ptr     = data_device;
    auto *       hmask_ptr  = qs_ptr + (QK_K / 4) * nblocks;
    auto *       scales_ptr = hmask_ptr + (QK_K / 8) * nblocks;
    sycl::half * d_ptr      = (sycl::half *) (scales_ptr + 12 * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        const block_q3_K * x  = (const block_q3_K *) tmp_buf;
        const int          ib = i;

        for (int j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }

        for (int j = 0; j < QK_K / 8; ++j) {
            hmask_ptr[ib * (QK_K / 8) + j] = x[ib].hmask[j];
        }

        for (int j = 0; j < 12; ++j) {
            scales_ptr[ib * 12 + j] = x[ib].scales[j];
        }

        d_ptr[ib] = x[ib].d;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

static bool reorder_qw_q5_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % sizeof(block_q5_K) == 0);
    GGML_ASSERT(offset % sizeof(block_q5_K) == 0);

    const int nblocks = size / sizeof(block_q5_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto * qs_ptr     = data_device;
    auto * qh_ptr     = qs_ptr + (QK_K / 2) * nblocks;
    auto * scales_ptr = qh_ptr + (QK_K / 8) * nblocks;
    auto * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        const block_q5_K * x  = (const block_q5_K *) tmp_buf;
        const int          ib = i;

        for (int j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }

        for (int j = 0; j < QK_K / 8; ++j) {
            qh_ptr[ib * (QK_K / 8) + j] = x[ib].qh[j];
        }

        for (int j = 0; j < K_SCALE_SIZE; ++j) {
            scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

static bool reorder_qw_q6_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % sizeof(block_q6_K) == 0);
    GGML_ASSERT(offset % sizeof(block_q6_K) == 0);

    const int nblocks = size / sizeof(block_q6_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto *       ql_ptr     = data_device;
    auto *       qh_ptr     = ql_ptr + (QK_K / 2) * nblocks;
    auto *       scales_ptr = qh_ptr + (QK_K / 4) * nblocks;
    sycl::half * dm_ptr     = (sycl::half *) (scales_ptr + (QK_K / 16) * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        const block_q6_K * x  = (const block_q6_K *) tmp_buf;
        const int          ib = i;

        const uint8_t * ql              = x[ib].ql;
        const uint8_t * qh              = x[ib].qh;
        uint8_t *       base_ql_ptr     = ql_ptr + (QK_K / 2) * ib;
        uint8_t *       base_qh_ptr     = qh_ptr + (QK_K / 4) * ib;
        uint8_t *       base_scales_ptr = scales_ptr + (QK_K / 16) * ib;

        for (int j = 0; j < QK_K / 2; ++j) {
            base_ql_ptr[j] = ql[j];
        }
        for (int j = 0; j < QK_K / 4; ++j) {
            base_qh_ptr[j] = qh[j];
        }

        for (int j = 0; j < QK_K / 16; ++j) {
            base_scales_ptr[j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].d;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

static bool reorder_qw(const ggml_tensor * src0, dpct::queue_ptr stream) {
    uint8_t * data_device = (uint8_t *) src0->data;
    size_t ncols = src0->ne[0];
    size_t nrows = src0->ne[1];
    size_t size = ggml_nbytes(src0);

    // MoE expert weights are addressed per expert via nb[2], so each slice must
    // remain self-contained after reorder.
    if (src0->ne[2] > 1) {
        GGML_ASSERT((size_t) size == (size_t) src0->ne[2] * src0->nb[2]);
        switch (src0->type) {
            case GGML_TYPE_Q4_K:
                return reorder_qw_q4_k_moe(data_device, src0->nb[2], src0->ne[2], stream);
            case GGML_TYPE_Q5_K:
                return reorder_qw_q5_k_moe(data_device, src0->nb[2], src0->ne[2], stream);
            case GGML_TYPE_Q6_K:
                return reorder_qw_q6_k_moe(data_device, src0->nb[2], src0->ne[2], stream);
            default:
                return false;
        }
    }

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            return reorder_qw_q4_0(data_device, ncols, nrows, size, 0, stream);
        case GGML_TYPE_Q8_0:
            return reorder_qw_q8_0(data_device, ncols, nrows, size, 0, stream);
        case GGML_TYPE_PQ2_0:
            return reorder_qw_pq2_0(data_device, ncols, nrows, size, 0, stream);
        case GGML_TYPE_PTQ1_0:
            return reorder_qw_ptq1_0(data_device, size, stream);
        case GGML_TYPE_Q2_K:
            return reorder_qw_q2_k(data_device, size, 0, stream);
        case GGML_TYPE_Q3_K:
            return reorder_qw_q3_k(data_device, size, 0, stream);
        case GGML_TYPE_Q4_K:
            return reorder_qw_q4_k(data_device, size, 0, stream);
        case GGML_TYPE_Q5_K:
            return reorder_qw_q5_k(data_device, size, 0, stream);
        case GGML_TYPE_Q6_K:
            return reorder_qw_q6_k(data_device, size, 0, stream);
        default:
            return false;
    }
}

static bool should_reorder_tensor(ggml_backend_sycl_context& ctx, const ggml_tensor * dst) {
    return g_ggml_sycl_enable_optimize && //allow optimize, controlled by $GGML_SYCL_ENABLE_OPT
           ctx.opt_feature.reorder &&      //allow this device due to good perf, skip the devices with bad perf.
           dst->op == GGML_OP_MUL_MAT &&   //limit to some supported cases of Q4_0, to do for more cases.
           // ne[1] <= 8 so multi-column decode (spec / MTP verify) also bootstraps the reorder;
           // all reorderable types have a _switch_ncols kernel.
           dst->src[1]->ne[1] <= 8 && dst->src[1]->ne[2]==1 && dst->src[1]->ne[3]==1;
}

static void opt_for_reorder(ggml_backend_sycl_context * ctx, const ggml_tensor * src0, const ggml_tensor * /* src1 */,
                            ggml_tensor * dst, mul_mat_algo mm_algorithm) {
    if (!should_reorder_tensor(*ctx, dst)) {
        return;
    }

    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    if (!extra || extra->optimized_feature.reorder) {
        return;  // Skip permutations and already reordered tensors
    }

    // PQ2_0/PTQ1_0 on 16-wide DPAS devices belong to the XMX layout (ggml_sycl_pq2_xmx_use) unless
    // ggml_sycl_xmx_disabled; it rewrites the same bytes in place, so reordering them as well would leave one path
    // reading the other's layout
    if ((src0->type == GGML_TYPE_PQ2_0 || src0->type == GGML_TYPE_PTQ1_0) &&
        (extra->optimized_feature.xmx_pq2 ||
         (!ggml_sycl_xmx_disabled(ctx->device) && ggml_sycl_device_has_dpas16(ctx->device)))) {
        return;
    }

    switch (mm_algorithm) {
        case mul_mat_algo::DMMV:
            if (!ggml_sycl_supports_reorder_dmmv(src0->type)) {
                return;
            }
            break;
        case mul_mat_algo::MMVQ:
            if (!ggml_sycl_supports_reorder_mmvq(src0->type)) {
                return;
            }
            break;
        case mul_mat_algo::MUL_MAT_SYCL:
            if (!ggml_sycl_supports_reorder_mul_mat_sycl(src0->type)) {
                return;
            }
            break;
    }

    if (reorder_qw(src0, ctx->stream())) {
        extra->optimized_feature.reorder = true;  // Used to decode/dequan in next steps and avoid re-reordering
    }
}

// Lazily reorder supported MoE expert weights once their fused path is used.
static void opt_for_reorder_id(ggml_backend_sycl_context * ctx, const ggml_tensor * src0) {
    if (!g_ggml_sycl_enable_optimize || !ctx->opt_feature.reorder) {
        return;
    }
    if (src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K) {
        return;
    }
    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    if (!extra || extra->optimized_feature.reorder) {
        return;
    }
    if (reorder_qw(src0, ctx->stream())) {
        extra->optimized_feature.reorder = true;
    }
}


static bool can_use_dequantize_mul_mat_vec(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    // The F16/BF16 qk=1 kernel iterates with stride 2*DMMV_X, requiring ne[0] to be
    // a multiple of 2*DMMV_X. Quantized types use block-structured kernels that only
    // need ne[0] % DMMV_X == 0.
    const int64_t dmmv_x_required = (src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F16) ?
                                    2*GGML_SYCL_DMMV_X : GGML_SYCL_DMMV_X;
    return ggml_sycl_supports_dmmv(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src0->ne[0] % dmmv_x_required == 0 && src1->ne[1] == 1;
}

static bool can_use_mul_mat_vec_q(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    return ggml_sycl_supports_mmvq(src0->type) &&
           src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;
}

// PQ2_0/PTQ1_0 weights on 16-wide DPAS devices are rewritten into the XMX layout on first use. From then on every
// mul_mat on them has to take that path, so the layout flag alone decides once it is set.
static bool ggml_sycl_pq2_xmx_use(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                  const ggml_tensor * dst) {
    if (src0->type != GGML_TYPE_PQ2_0 && src0->type != GGML_TYPE_PTQ1_0) {
        return false;
    }
    // MUL_MAT_ID passes each expert as a 2D copy of the 3D weight that shares its extra, and a view shares
    // its parent's data: rewriting either in place would corrupt the rest of the tensor
    if (dst->op != GGML_OP_MUL_MAT || src0->view_src != nullptr) {
        return false;
    }
    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    if (extra && extra->optimized_feature.xmx_pq2) {
        return true;
    }
    // already in another in-place layout (MMVQ reorder or the TernSYCL 2-bit repack): rewriting it would corrupt it
    if (extra && (extra->optimized_feature.reorder || extra->optimized_feature.t2)) {
        return false;
    }
    if (!g_ggml_sycl_enable_optimize || ggml_sycl_xmx_disabled(ctx.device) ||
        !ggml_sycl_device_has_dpas16(ctx.device)) {
        return false;
    }
    // op offload refills COMPUTE buffers from host memory every time, so an in-place layout there goes stale
    if (!extra || ggml_backend_buffer_is_sycl_split(src0->buffer) ||
        src0->buffer->usage == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || !ggml_is_contiguous(src0) ||
        !ggml_sycl_pq2_xmx_supports_ne0(src0->ne[0]) || (uintptr_t) src0->data % 64 != 0) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || src1->nb[0] != sizeof(float) || dst->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(dst)) {
        return false;
    }
    // PTQ1_0 expands to 34 bytes a block, which only fits where the buffer reserved room for it
    if (src0->type == GGML_TYPE_PTQ1_0 &&
        ggml_backend_buft_get_alloc_size(src0->buffer->buft, src0) < ggml_sycl_ptq1_xmx_bytes(src0)) {
        return false;
    }
    if (!ggml_sycl_pq2_xmx_reorder(const_cast<ggml_tensor *>(src0), ctx.stream())) {
        return false;
    }
    extra->optimized_feature.xmx_pq2 = true;
    return true;
}

// ARC-LAB: small-output bf16 mat-mul for 2..8 activation columns (speculative-decoding verify batches through Bonsai's
// delta-net gate projections, 48 x 5120). The generic path converted the whole bf16 matrix and ran a GEMM per call.
// One 256-wide work-group per output row reads the row once (8 bf16 per load) and dots it with every column.
static bool ggml_sycl_bf16_small_ok(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    static const bool off = getenv("GGML_SYCL_BF16_ROWSPLIT_OFF") != nullptr;
    return !off && src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src0->ne[1] <= 512 && src1->ne[1] >= 2 && src1->ne[1] <= 8 && src0->ne[0] % 8 == 0 &&
           src1->ne[2] * src1->ne[3] == 1 && src0->ne[2] * src0->ne[3] == 1 &&
           ggml_is_contiguous(src0) && ggml_is_contiguous_rows(src1) && ggml_is_contiguous(dst) &&
           ((uintptr_t) src0->data % 16) == 0 && !ggml_backend_buffer_is_sycl_split(src0->buffer);
}

static void ggml_sycl_mul_mat_bf16_small(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    constexpr int WG = 256;
    const int ncols_x = src0->ne[0];
    const int nrows   = src0->ne[1];
    const int ncols_y = src1->ne[1];
    const int64_t sy  = src1->nb[1] / sizeof(float);
    const int64_t sd  = dst->nb[1] / sizeof(float);
    const uint16_t * x = (const uint16_t *) src0->data;
    const float *    y = (const float *) src1->data;
    float *          d = (float *) dst->data;
    ctx.stream()->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) nrows * WG), sycl::range<1>(WG)),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            const int row = it.get_group(0);
            const int tid = it.get_local_id(0);
            const uint16_t * xr = x + (size_t) row * ncols_x;
            float sum[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
            for (int i = tid * 8; i < ncols_x; i += WG * 8) {
                const sycl::vec<uint16_t, 8> w = *(const sycl::vec<uint16_t, 8> *) (xr + i);
                float wf[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    wf[j] = sycl::bit_cast<float>((uint32_t) w[j] << 16);
                }
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    if (c < ncols_y) {
                        const float * yc = y + c * sy + i;
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            sum[c] += wf[j] * yc[j];
                        }
                    }
                }
            }
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                if (c < ncols_y) {
                    const float v = sycl::reduce_over_group(it.get_group(), sum[c], sycl::plus<float>());
                    if (tid == 0) {
                        d[c * sd + row] = v;
                    }
                }
            }
        });
}

// ARC-LAB: PTQ1_0 weights chosen by GGML_SYCL_PTQ1_T2 are repacked in place (their buffer was sized for it) on first use
// and from then on only this path may read them.
static bool ggml_sycl_t2_tensor(const ggml_tensor * w) {
    if ((w->type != GGML_TYPE_PTQ1_0 && w->type != GGML_TYPE_PQ2_0) || w->ne[2] != 1 || w->ne[3] != 1 || !w->extra) {
        return false;
    }
    const auto * extra = static_cast<const ggml_tensor_extra_gpu *>(w->extra);
    return extra->optimized_feature.t2 || ggml_sycl_t2_wants(w->name, w->ne[0], w->ne[1]);
}

static bool ggml_sycl_t2_mul_mat_try(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                     ggml_tensor * dst) {
    if (!ggml_sycl_t2_tensor(src0)) {
        return false;
    }
    auto * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    GGML_ASSERT(!extra->optimized_feature.reorder && "PTQ1_0 T2 tensor was reordered");
    const int64_t K = src0->ne[0], N = src0->ne[1];
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && src1->ne[0] == K && src1->nb[0] == sizeof(float));
    GGML_ASSERT(ggml_is_contiguous(dst) && dst->ne[0] == N);
    GGML_ASSERT(src1->ne[3] == 1 && (src1->ne[2] == 1 || src1->nb[2] == src1->ne[1] * src1->nb[1]));
    dpct::queue_ptr stream = ctx.stream();
    if (!extra->optimized_feature.t2) {
        if (!ggml_sycl_t2_repack(*stream, src0->data, K, N, src0->type == GGML_TYPE_PQ2_0)) {
            GGML_ABORT("PQ2_0 tensor %s uses code 3 (+2), which the 2-bit XMX layout cannot hold: unset GGML_SYCL_PTQ1_T2",
                       src0->name);
        }
        extra->optimized_feature.t2 = true;
    }
    // tokens: ne[1] x ne[2] rows of K floats (row stride nb[1]; planes back to back)
    const int64_t M = src1->ne[1] * src1->ne[2];
    // ARC-LAB: large prompt batches through oneDNN's int8 GEMM (faster than GemmMT there), GGML_SYCL_T2_W8A8_MIN (0 = off)
    static const int64_t w8a8_env = getenv("GGML_SYCL_T2_W8A8_MIN") ? atoll(getenv("GGML_SYCL_T2_W8A8_MIN")) : 0;
    const int64_t        w8a8_min = ggml_sycl_knob("GGML_SYCL_T2_W8A8_MIN", (int) w8a8_env);
    if (w8a8_min > 0 && M >= w8a8_min && g_ggml_sycl_enable_dnn && src1->nb[1] == (size_t) K * sizeof(float) &&
        ggml_sycl_w8a8_mul_mat_t2(ctx, src0->data, (const float *) src1->data, K, (float *) dst->data, N, M, K, stream)) {
        return true;
    }
    ggml_sycl_t2_mul_mat(*stream, src0->data, (const float *) src1->data, src1->nb[1] / sizeof(float), (float *) dst->data,
                         M, N, K, src1->view_src && src1->view_offs == 0 ? src1->view_src : src1);
    return true;
}

// ARC-LAB: largest batch run as MMVQ column chunks instead of dequantize-all + GEMM (GGML_SYCL_MMVQ_CHUNK_MAX, default 32).
// A chunk re-reads the quantized weights; the GEMM path writes and reads the whole matrix as f16 per call, so chunks win
// for speculative-verify sized batches (Gemma 4 12B q4_0: n-gram drafts of 33+ tokens were ~94% dequantization).
static int64_t mmvq_chunk_max() {
    static const int64_t v = getenv("GGML_SYCL_MMVQ_CHUNK_MAX") ? atoll(getenv("GGML_SYCL_MMVQ_CHUNK_MAX")) : 4 * MMVQ_MAX_BATCH_SIZE;
    return v;
}

static void ggml_sycl_mul_mat(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    if (dst->op == GGML_OP_MUL_MAT && ggml_sycl_t2_mul_mat_try(ctx, src0, src1, dst)) {
        return;
    }

    // Handle HADAMARAD hint given from further up the pipeline and pass it to the correct
    // kernel.
    //
    // The op check is not redundant: this backend also routes MUL_MAT_ID through here with a
    // stack copy of dst, which carries MUL_MAT_ID's own op_params. ggml_mul_mat_set_hint()
    // asserts GGML_OP_MUL_MAT for the same reason.
    if (dst->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
        ggml_sycl_op_fwht(ctx, src1, dst)) {
        return;
    }

    if (ggml_sycl_pq2_xmx_use(ctx, src0, src1, dst)) {
        ggml_sycl_pq2_xmx_mul_mat(ctx, src0, src1, dst);
        return;
    }

    const bool split = ggml_backend_buffer_is_sycl_split(src0->buffer);
    int64_t min_compute_capability = INT_MAX;

    if (split) {
        ggml_backend_sycl_split_buffer_type_context * buft_ctx =
            (ggml_backend_sycl_split_buffer_type_context *) src0->buffer->buft->context;
        auto & tensor_split = buft_ctx->tensor_split;
        for (int id = 0; id < ggml_sycl_info().device_count; ++id) {
            // skip devices that are not going to do any work:
            if (tensor_split[id] >= (id + 1 < ggml_sycl_info().device_count ? tensor_split[id + 1] : 1.0f)) {
                continue;
            }

            if (min_compute_capability > ggml_sycl_info().devices[id].cc) {
                min_compute_capability = ggml_sycl_info().devices[id].cc;
            }
        }
    } else {
        min_compute_capability = ggml_sycl_info().devices[ctx.device].cc;
    }

    // ARC-LAB: reordered q4_0 with 5..64 columns on XMX (weights streamed once for all columns), GGML_SYCL_Q4_0_DPAS=1
    {
        static const bool q4d_on = getenv("GGML_SYCL_Q4_0_DPAS") && atoi(getenv("GGML_SYCL_Q4_0_DPAS")) != 0 &&
                                   ggml_sycl_device_is_xe2();
        static const int  q4d_min = getenv("GGML_SYCL_Q4_0_DPAS_MIN") ? atoi(getenv("GGML_SYCL_Q4_0_DPAS_MIN")) : 5;
        static const int  q4d_max = getenv("GGML_SYCL_Q4_0_DPAS_MAX") ? atoi(getenv("GGML_SYCL_Q4_0_DPAS_MAX")) : 256;
        const auto * ex = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
        if (q4d_on && !split && src0->type == GGML_TYPE_Q4_0 && ex && ex->optimized_feature.reorder &&
            ggml_is_contiguous(src0) && src0->ne[2] == 1 && src0->ne[3] == 1 && src1->type == GGML_TYPE_F32 &&
            dst->type == GGML_TYPE_F32 && src1->ne[2] == 1 && src1->ne[3] == 1 && src1->nb[0] == sizeof(float) &&
            src1->ne[1] >= q4d_min && src1->ne[1] <= q4d_max && ggml_is_contiguous(dst) &&
            ggml_sycl_q4_0_dpas_gemm(ctx, src0->data, (const float *) src1->data, src1->nb[1] / (int64_t) sizeof(float),
                                     (float *) dst->data, (int) src0->ne[1], (int) src0->ne[0], (int) src1->ne[1], ctx.stream())) {
            return;
        }
    }

    // Small batches (e.g. speculative verify) on quantized weights: run MMVQ on chunks of columns
    // instead of dequantizing all of src0 to fp32 (the output matrix is ~5 GB as fp32).
    if (!split && ggml_is_quantized(src0->type) && ggml_is_contiguous(src0) && src0->ne[2] == 1 && src0->ne[3] == 1 &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        src1->ne[1] > MMVQ_MAX_BATCH_SIZE && src1->nb[0] == sizeof(float) &&
        (src1->ne[1] <= mmvq_chunk_max() || (size_t) src0->ne[0] * src0->ne[1] * sizeof(float) > (1ull << 30))) {
        for (int64_t c0 = 0; c0 < src1->ne[1]; c0 += MMVQ_MAX_BATCH_SIZE) {
            const int64_t nc = std::min<int64_t>(MMVQ_MAX_BATCH_SIZE, src1->ne[1] - c0);
            ggml_tensor src1_c = *src1;
            ggml_tensor dst_c  = *dst;
            src1_c.ne[1] = nc;
            src1_c.data  = (char *) src1->data + c0 * src1->nb[1];
            src1_c.nb[2] = src1_c.nb[3] = nc * src1->nb[1];
            dst_c.ne[1]  = nc;
            dst_c.data   = (char *) dst->data + c0 * dst->nb[1];
            dst_c.nb[2]  = dst_c.nb[3] = nc * dst->nb[1];
            dst_c.src[1] = &src1_c;
            ggml_sycl_mul_mat(ctx, src0, &src1_c, &dst_c);
        }
        return;
    }

    // check data types and tensor shapes for custom matrix multiplication kernels:
    bool use_dequantize_mul_mat_vec = can_use_dequantize_mul_mat_vec(src0, src1, dst);

    bool use_mul_mat_vec_q = can_use_mul_mat_vec_q(src0, src1, dst);

    bool use_mul_mat_q =  ggml_sycl_supports_mmq(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;


    // mmvq and mmq need the __dp4a instruction which is available for gen12+
    // Workaround in https://github.com/ggml-org/llama.cpp/commit/95f84d5ce8b449a9b16009434aca800df504a02e
    use_mul_mat_q = use_mul_mat_q && (src0->type != GGML_TYPE_IQ2_XXS);
#ifdef SYCL_USE_XMX
    use_mul_mat_q = use_mul_mat_q && (src1->ne[1] <= MMQ_MAX_BATCH_SIZE);
#endif // SYCL_USE_XMX

    // When reorder is enabled, both ESIMD, MMVQ and DMMV kernels may be used. For
    // best performance use ESIMD when supported, followed by MMVQ, and finally DMMV.
    // But the reordered ESIMD path cannot be used without reordered MMVQ. A later
    // multi-token call (ne[1] in 2..8) will take the MMVQ path and it would read the
    // reordered bytes as if they were still the unreordered layout.

    if (!g_ggml_sycl_prioritize_dmmv && ((should_reorder_tensor(ctx, dst) &&
                                          ggml_sycl_supports_reorder_mmvq(src0->type)))) {
        bool use = g_ggml_sycl_enable_esimd && ggml_sycl_supports_reorder_esimd(src0->type);
        // Arc770 get benefit with Q4_0 by skipping MMVQ path
        if (!(ggml_sycl_info().devices[ctx.device].hw_info.arch ==
                    gpu_arch::intel_gpu_acm_g10 &&
                src0->type == GGML_TYPE_Q4_0)) {
            use = use || !use_mul_mat_vec_q;
        }
        use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && use;
    }

    if (!split && src0->type == GGML_TYPE_F16 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && src1->ne[1] == 1) {
        // TODO: Refactor and cleanup of mul mat dispatching.
        if (src0->ne[3] == 1 && src1->ne[3] == 1) {
            // KQ single-batch
            // mmv p021 was specific for these dimensions
            ggml_sycl_mul_mat_vec_p021(ctx, src0, src1, dst);
        } else {
            // The kernel from the if path is faster for that specific case, but does not support all mul mats.
            ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
        }
    } else if (!split && src0->type == GGML_TYPE_F16 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && src1->ne[1] == 1 && src1->ne[3] == 1) {
        // KQV single-batch
        ggml_sycl_mul_mat_vec_nc(ctx, src0, src1, dst);
    } else if (!split && src0->type == GGML_TYPE_F16 && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2] * src1->ne[3] > 1) {
        // KQ + KQV multi-batch
        ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
    } else if (!split && ggml_sycl_bf16_small_ok(src0, src1, dst)) {
        ggml_sycl_mul_mat_bf16_small(ctx, src0, src1, dst);
    } else if (use_dequantize_mul_mat_vec) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::DMMV);
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_dequantize_mul_mat_vec);
    } else if (use_mul_mat_vec_q) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::MMVQ);
        ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
        if (extra && extra->optimized_feature.reorder) {
            if (src0->type == GGML_TYPE_PTQ1_0 && ggml_sycl_ptq1_0_ncols_dec_enabled() && src1->ne[1] >= 2 && src1->ne[1] <= 8) {
                // ARC-LAB: decode-once PTQ1_0 multi-column kernel reads the chunk-interleaved activation layout
                ggml_sycl_op_mul_mat<quantize_and_reorder_q8_1_ptq1_il>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
            } else {
                ggml_sycl_op_mul_mat<quantize_and_reorder_q8_1_soa>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
            }
        } else {
            ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
        }
    } else if (use_mul_mat_q) {
        ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_q);
    } else {
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_sycl);
    }
}

// Fused dense-FFN mat-vec for the {mul_mat(gate), mul_mat(up), GLU} subgraph at node_idx.
// Returns false if it declined, in which case the caller runs the three nodes normally.
static bool ggml_sycl_mul_mat_glu_mmvq_fused(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx) {
    static const bool fuse_off = getenv("GGML_SYCL_GLU_FUSE_OFF") != nullptr;  // ARC-LAB A/B switch
    if (fuse_off || !ggml_sycl_can_fuse(cgraph, node_idx, { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU }, {})) {
        return false;
    }
    if (ggml_sycl_t2_tensor(cgraph->nodes[node_idx]->src[0]) || ggml_sycl_t2_tensor(cgraph->nodes[node_idx + 1]->src[0])) {
        return false;  // ARC-LAB: 2-bit repacked weights (ptq1-t2.cpp) run unfused
    }

    ggml_tensor *       glu  = cgraph->nodes[node_idx + 2];
    ggml_tensor *       gate = glu->src[0];
    ggml_tensor *       up   = glu->src[1];
    const ggml_tensor * wu   = up->src[0];
    const ggml_tensor * wg   = gate->src[0];
    const ggml_tensor * act  = up->src[1];

    // this writes glu->data directly rather than the per-device row slices that
    // ggml_sycl_op_mul_mat() stitches back together, so it cannot serve split weights
    if (ggml_backend_buffer_is_sycl_split(wu->buffer) || ggml_backend_buffer_is_sycl_split(wg->buffer)) {
        return false;
    }

    // with DMMV prioritised the unfused path would not have gone through mmvq at all
    if (g_ggml_sycl_prioritize_dmmv) {
        return false;
    }

    // install the reorder (SoA) layout the fused kernel needs, as the unfused mmvq path would;
    // a no-op once done. after the bail checks so a declined op does not pay for it.
    opt_for_reorder(&ctx, wu, act, up, mul_mat_algo::MMVQ);
    opt_for_reorder(&ctx, wg, act, gate, mul_mat_algo::MMVQ);

    const auto * extra_u = static_cast<const ggml_tensor_extra_gpu *>(wu->extra);
    const auto * extra_g = static_cast<const ggml_tensor_extra_gpu *>(wg->extra);
    if (!extra_u || !extra_g || !extra_u->optimized_feature.reorder || !extra_g->optimized_feature.reorder) {
        return false;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, up, /*num_src=*/2, " : fused with gate + GLU");

    const int64_t ne00 = wu->ne[0];
    const int64_t ne11 = act->ne[1];

    const queue_ptr stream           = ctx.stream();
    const int       src1_padded_cols = GGML_PAD((int) ne00, MATRIX_ROW_PADDING);

    // one activation, quantized once and fully consumed into src1_ddq before the GEMV on this
    // in-order queue, so glu->data aliasing the dead activation needs no memory-range check
    ggml_sycl_pool_alloc<char> src1_q8_alloc(ctx.pool(),
                                             (size_t) ne11 * src1_padded_cols * sizeof(block_q8_1) / QK8_1);
    char *                     src1_ddq = src1_q8_alloc.get();

    if (wu->type == GGML_TYPE_PTQ1_0 && ggml_sycl_ptq1_0_ncols_dec_enabled() && ne11 <= 8 &&
        !(ne11 == 1 && ggml_sycl_ptq1_0_glu_n1_enabled())) {
        // ARC-LAB: the decode-once PTQ1_0 fused kernel reads the chunk-interleaved activation layout
        quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_ptq1_il>((const float *) act->data, src1_ddq, (int) ne00,
                                                                  (int) ne11, src1_padded_cols, stream);
    } else {
        quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>((const float *) act->data, src1_ddq, (int) ne00, (int) ne11,
                                                              src1_padded_cols, stream);
    }

    return ggml_sycl_mul_mat_vec_q_glu_reorder(wu->type, ggml_get_glu_op(glu), wu->data, wg->data, src1_ddq,
                                               (float *) glu->data, (int) ne00, (int) wu->ne[1], (int) ne11,
                                               /*stride_col_y_bytes=*/src1_padded_cols * (int) sizeof(block_q8_1) /
                                                   QK8_1,
                                               /*stride_col_dst=*/(int) glu->ne[0], stream);
}

__dpct_inline__ static void k_copy_src1_to_contiguous(
    const char *__restrict__ src1_original, char *__restrict__ src1_contiguous,
    const mmid_row_mapping *__restrict__ row_mapping,
    int64_t ne11, int64_t ne10, size_t nb11, size_t nb12,
    const sycl::nd_item<3> &item_ct1) {
    const int32_t src1_row = item_ct1.get_group(2);

    const int32_t iid1 = row_mapping[src1_row].i2;
    const int32_t id   = row_mapping[src1_row].i1;

    const int64_t i11 = id % ne11;
    const int64_t i12 = iid1;

    const float * src1_row_original = (const float *)(src1_original + i11*nb11 + i12*nb12);
    float * src1_row_contiguous = (float *)(src1_contiguous + src1_row*nb11);

#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < ne10;
         i += item_ct1.get_local_range(2)) {
        src1_row_contiguous[i] = src1_row_original[i];
    }
}

__dpct_inline__ static void k_copy_dst_from_contiguous(
    char *__restrict__ dst_original, const char *__restrict__ dst_contiguous,
    const mmid_row_mapping *__restrict__ row_mapping, int64_t ne0, size_t nb1,
    size_t nb2, const sycl::nd_item<3> &item_ct1) {
    int32_t i = item_ct1.get_group(2);

    const int32_t i1 = row_mapping[i].i1;
    const int32_t i2 = row_mapping[i].i2;

    const float * dst_row_contiguous = (const float *)(dst_contiguous + i*nb1);
    float * dst_row_original = (float *)(dst_original + i1*nb1 + i2*nb2);

#pragma unroll
    for (int j = item_ct1.get_local_id(2); j < ne0;
         j += item_ct1.get_local_range(2)) {
        dst_row_original[j] = dst_row_contiguous[j];
    }
}

// Fused MoE TG fast path. Returns false to fall back to the per-expert loop below.
static bool ggml_sycl_mul_mat_id_mmvq_fused(
    ggml_backend_sycl_context & ctx, const ggml_tensor * src0,
    const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst)
{
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    if (ne12 != 1) return false;
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (ne10 != src0->ne[0] || ne10 % QK8_1 != 0) return false;
    if (!ggml_is_contiguous(src1)) return false;

    const int64_t n_ids_per_group = ids->ne[0];
    if (ids->ne[1] != 1) return false;
    if (ne11 != 1 && ne11 != n_ids_per_group) return false;

    const queue_ptr stream           = ctx.stream();
    const int       src1_padded_cols = GGML_PAD((int) ne10, MATRIX_ROW_PADDING);
    const int       n_experts_used   = (int) n_ids_per_group;
    const int       nrows            = (int) src0->ne[1];

    // Lazily reorder the (Q4_K) expert weights into a per-expert SoA layout, then run the reorder
    // GEMV. Placed after the bail checks so a non-dispatchable op does not pay the reorder cost.
    opt_for_reorder_id(&ctx, src0);
    const ggml_tensor_extra_gpu * src0_extra =
        static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    const bool use_reorder = src0_extra && src0_extra->optimized_feature.reorder;

    ggml_sycl_pool_alloc<char> src1_q8_alloc(ctx.pool(),
        (size_t) ne11 * src1_padded_cols * sizeof(block_q8_1) / QK8_1);
    char * src1_ddq = src1_q8_alloc.get();
    if (use_reorder) {
        quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(
            (const float *) src1->data, src1_ddq, (int) ne10, (int) ne11,
            src1_padded_cols, stream);
    } else {
        quantize_row_q8_1_sycl<quantize_q8_1>(
            (const float *) src1->data, src1_ddq, (int) ne10, (int) ne11,
            src1_padded_cols, stream);
    }

    const size_t bytes_per_qrow = (size_t) src1_padded_cols * sizeof(block_q8_1) / QK8_1;
    const size_t src1_row_stride = (ne11 == 1) ? 0 : bytes_per_qrow;

    if (use_reorder) {
        return ggml_sycl_mul_mat_vec_q_id_reorder(
            src0->type, src0->data, src1_ddq, (const int32_t *) ids->data,
            (float *) dst->data, (int) ne10, nrows, n_experts_used,
            /*expert_weight_stride=*/ src0->nb[2],
            /*dst_row_stride=*/ dst->nb[1],
            src1_row_stride, stream);
    }
    return ggml_sycl_mul_mat_vec_q_id(
        src0->type, src0->data, src1_ddq, (const int32_t *) ids->data,
        (float *) dst->data, (int) ne10, nrows, n_experts_used,
        /*expert_weight_stride=*/ src0->nb[2],
        /*dst_row_stride=*/ dst->nb[1],
        src1_row_stride, stream);
}

// counting sort of the routed rows by expert id (row_id_i, as chosen by the router):
// builds a projection of a memory layout where each expert's slice is contiguous
static void mmid_counting_sort_rows(
        const ggml_tensor * ids, const char * ids_host,
        int64_t n_ids, int64_t n_as, int64_t n_routed_rows,
        std::vector<int64_t> & expert_counts,
        std::vector<int64_t> & expert_row_offsets,
        std::vector<mmid_row_mapping> & routed_row_src) {

    // frequencies: how many routed rows each expert "owns"
    expert_counts.assign(n_as, 0);
    for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        for (int64_t id = 0; id < n_ids; id++) {
            const int32_t row_id_i = *(const int32_t *) (ids_host + iid1*ids->nb[1] + id*ids->nb[0]);
            GGML_ASSERT(row_id_i >= 0 && row_id_i < n_as);
            expert_counts[row_id_i]++;
        }
    }

    // where each expert's slice starts (row indices) and the previous ends
    expert_row_offsets.assign(n_as + 1, 0);
    for (int64_t i02 = 0; i02 < n_as; i02++) {
        expert_row_offsets[i02 + 1] = expert_row_offsets[i02] + expert_counts[i02];
    }

    std::vector<int64_t> expert_row_next = expert_row_offsets;
    routed_row_src.resize(n_routed_rows);
    for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        for (int64_t id = 0; id < n_ids; id++) {
            const int32_t row_id_i = *(const int32_t *) (ids_host + iid1*ids->nb[1] + id*ids->nb[0]);
            GGML_ASSERT(row_id_i >= 0 && row_id_i < n_as);

            // find and validate the next free row for a given expert (row_id_i)
            const int64_t routed_row = expert_row_next[row_id_i]++;
            GGML_ASSERT(routed_row >= expert_row_offsets[row_id_i]);
            GGML_ASSERT(routed_row < expert_row_offsets[row_id_i + 1]);
            routed_row_src[routed_row] = {(int32_t) id, (int32_t) iid1};
        }
    }
}

static void ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx,
                                 ggml_tensor *dst) try {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    const ggml_tensor *src0 = dst->src[0];
    const ggml_tensor *src1 = dst->src[1];
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer) && "mul_mat_id does not support split buffers");

    const ggml_tensor *ids = dst->src[2];
    GGML_TENSOR_BINARY_OP_LOCALS

    const queue_ptr stream = ctx.stream();

    const int64_t n_as = ne02;
    const int64_t n_ids = ids->ne[0];

    if (ne12 == 1) {
        if (ggml_sycl_mul_mat_id_mmvq_fused(ctx, src0, src1, ids, dst)) {
            return;
        }
    }

    std::vector<char> ids_host(ggml_nbytes(ids));
    const char * ids_dev = (const char *) ids->data;

    SYCL_CHECK(CHECK_TRY_ERROR(
        stream->memcpy(ids_host.data(), ids_dev, ggml_nbytes(ids))));

    // also ensures ctx.mmid_row_mapping_host is drained before we use it again
    SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));

    ggml_tensor src0_row = *src0;
    ggml_tensor src1_row = *src1;
    ggml_tensor dst_row = *dst;

    char *src0_original = (char *)src0->data;
    char *src1_original = (char *)src1->data;
    char *dst_original = (char *)dst->data;

    src0_row.ne[2] = 1;
    src0_row.ne[3] = 1;
    src0_row.nb[3] = nb02;

    src1_row.ne[1] = 1;
    src1_row.ne[2] = 1;
    src1_row.ne[3] = 1;
    src1_row.nb[2] = nb11;
    src1_row.nb[3] = nb11;

    dst_row.ne[1] = 1;
    dst_row.ne[2] = 1;
    dst_row.ne[3] = 1;
    dst_row.nb[2] = nb1;
    dst_row.nb[3] = nb1;
    if (ne12 == 1) {
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
            for (int64_t id = 0; id < n_ids; id++) {
                const int32_t i02 = *(const int32_t *) (ids_host.data() + iid1*ids->nb[1] + id*ids->nb[0]);
                GGML_ASSERT(i02 >= 0 && i02 < n_as);

                const int64_t i11 = id % ne11;
                const int64_t i12 = iid1;

                const int64_t i1 = id;
                const int64_t i2 = i12;

            src0_row.data = src0_original + i02*nb02;
            src1_row.data = src1_original + i11*nb11 + i12*nb12;
            dst_row.data = dst_original + i1*nb1 + i2*nb2;

            ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_row);
            }
        }
    } else {
        const int64_t n_routed_rows = ids->ne[1] * n_ids;
        ggml_sycl_pool_alloc<char> src1_contiguous(ctx.pool(), sizeof(float)*n_routed_rows*ne10);
        ggml_sycl_pool_alloc<char>  dst_contiguous(ctx.pool(), sizeof(float)*n_routed_rows*ne0);

        src1_row.data = src1_contiguous.get();
        dst_row.data  =  dst_contiguous.get();

        // how many "owned" routed rows to pass to each expert
        std::vector<int64_t> expert_row_counts;
        // where each expert's slice starts and the previous ends (row indices, right-exclusive)
        std::vector<int64_t> expert_row_offsets;
        // the sources (slot/token pairs) of contiguous rows to guide k_copy_src1_to_contiguous
        std::vector<mmid_row_mapping> & routed_row_src = ctx.mmid_row_mapping_host;

        mmid_counting_sort_rows(ids, ids_host.data(), n_ids, n_as, n_routed_rows,
                                expert_row_counts, expert_row_offsets, routed_row_src);

        ggml_sycl_pool_alloc<mmid_row_mapping> dev_row_mapping(ctx.pool(), n_routed_rows);
        SYCL_CHECK(CHECK_TRY_ERROR(
                stream->memcpy(dev_row_mapping.get(), routed_row_src.data(), n_routed_rows*sizeof(mmid_row_mapping))));

        const unsigned int max_work_group_size = ggml_sycl_info().max_work_group_sizes[ctx.device];
        assert(max_work_group_size % (WARP_SIZE * WARP_SIZE) == 0);

        {
            sycl::range<3> block_dims(1, 1, std::min((unsigned int)ne10, max_work_group_size));
            sycl::range<3> grid_dims(1, 1, n_routed_rows);
            stream->submit([&](sycl::handler &cgh) {
                char *__restrict src1_contiguous_get =
                    src1_contiguous.get();
                mmid_row_mapping *__restrict dev_row_mapping_get =
                    dev_row_mapping.get();

                cgh.parallel_for(
                    sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_copy_src1_to_contiguous(
                            src1_original, src1_contiguous_get,
                            dev_row_mapping_get,
                            ne11, ne10, nb11, nb12,
                            item_ct1);
                    });
            });
        }

        for (int64_t i02 = 0; i02 < n_as; i02++) {
            const int64_t num_src1_rows = expert_row_counts[i02];

            if (num_src1_rows == 0) {
                continue;
            }

            const int64_t expert_row_offset = expert_row_offsets[i02];

            src0_row.data = src0_original + i02*nb02;

            GGML_ASSERT(nb11 == sizeof(float)*ne10);
            GGML_ASSERT(nb1 == sizeof(float)*ne0);
            src1_row.data = src1_contiguous.get() + expert_row_offset*nb11;
            src1_row.ne[1] = num_src1_rows;

            src1_row.nb[1] = nb11;
            src1_row.nb[2] = num_src1_rows*nb11;
            src1_row.nb[3] = num_src1_rows*nb11;

            dst_row.data = dst_contiguous.get() + expert_row_offset*nb1;
            dst_row.ne[1] = num_src1_rows;
            dst_row.nb[1] = nb1;
            dst_row.nb[2] = num_src1_rows*nb1;
            dst_row.nb[3] = num_src1_rows*nb1;

            ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_row);
        }

        {
            sycl::range<3> block_dims(1, 1, std::min((unsigned int)ne0, max_work_group_size));
            sycl::range<3> grid_dims(1, 1, n_routed_rows);
            stream->submit([&](sycl::handler &cgh) {
                const char *__restrict dst_contiguous_get =
                    dst_contiguous.get();
                const mmid_row_mapping *__restrict dev_row_mapping_get =
                    dev_row_mapping.get();

                cgh.parallel_for(
                    sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_copy_dst_from_contiguous(dst_original,
                                                   dst_contiguous_get,
                                                   dev_row_mapping_get,
                                                   ne0, nb1, nb2, item_ct1);
                    });
            });
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_sycl_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_scale(ctx, dst);
}

static void ggml_sycl_diag_mask_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_diag_mask_inf(ctx, dst);
}

static void ggml_sycl_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pool2d(ctx, dst);
}

static void ggml_sycl_pool1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pool1d(ctx, dst);
}

static void ggml_sycl_im2col(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_im2col(ctx, dst);
}

static void ggml_sycl_im2col_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_im2col_3d(ctx, dst);
}

static void ggml_sycl_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_col2im_1d(ctx, dst);
}

static void ggml_sycl_conv_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_conv_3d(ctx, dst);
}

static void ggml_sycl_sum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_sum(ctx, dst);
}

static void ggml_sycl_sum_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_sum_rows(ctx, dst);
}

static void ggml_sycl_mean(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_mean(ctx, dst);
}

static void ggml_sycl_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_argsort(ctx, dst);
}

static void ggml_sycl_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_argmax(ctx, dst);
}


static void ggml_sycl_set_main_device(const int main_device) try {
    if (dpct::get_current_device_id() == static_cast<unsigned int> (main_device)) {
        return;
    }
    check_allow_gpu_index(main_device);
    dpct::select_device(main_device);

    if (g_ggml_sycl_debug) {
        dpct::device_info prop;
        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, dpct::dev_mgr::instance().get_device(main_device))));
        GGML_LOG_INFO("Using device %d (%s) as main device\n",
                main_device, prop.get_name());
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static bool ggml_sycl_compute_forward(ggml_backend_sycl_context & ctx, struct ggml_tensor * dst) try {
    if (!g_sycl_loaded) return false;

    if (dst->src[0] != nullptr && ggml_backend_buffer_is_sycl_split(dst->src[0]->buffer)) {
        ggml_sycl_set_peer_access(dst->src[1]->ne[1], ctx.device);
    }

    switch (dst->op) {
        case GGML_OP_ARGMAX:
            ggml_sycl_argmax(ctx, dst);
            break;
        case GGML_OP_CONV_2D:
            ggml_sycl_op_conv2d(ctx, dst);
            break;
        case GGML_OP_CONV_2D_DW:
            ggml_sycl_op_conv2d_dw(ctx, dst);
            break;
        case GGML_OP_CONV_3D:
            ggml_sycl_conv_3d(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_sycl_op_conv_transpose_1d(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            ggml_sycl_op_conv2d_transpose(ctx, dst);
            break;
        case GGML_OP_REPEAT:
            ggml_sycl_repeat(ctx, dst);
            break;
        case GGML_OP_REPEAT_BACK:
            ggml_sycl_repeat_back(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_sycl_get_rows(ctx, dst);
            break;
        case GGML_OP_SET:
            ggml_sycl_op_set(ctx, dst);
            break;
        case GGML_OP_SET_ROWS:
            ggml_sycl_op_set_rows(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_PRE:
            ggml_sycl_op_dsv4_hc_pre(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_COMB:
            ggml_sycl_op_dsv4_hc_comb(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_POST:
            ggml_sycl_op_dsv4_hc_post(ctx, dst);
            break;
        case GGML_OP_LIGHTNING_INDEXER:
            ggml_sycl_op_lightning_indexer(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_sycl_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
        case GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_sycl_add(ctx, dst);
            break;
        case GGML_OP_ADD_ID:
            ggml_sycl_add_id(ctx, dst);
            break;
        case GGML_OP_SUB:
            ggml_sycl_sub(ctx, dst);
            break;
        case GGML_OP_COUNT_EQUAL:
            ggml_sycl_count_equal(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_sycl_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_sycl_mul(ctx, dst);
            break;
        case GGML_OP_LOG:
            ggml_sycl_log(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_sycl_div(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_NEG:
                    ggml_sycl_neg(ctx, dst);
                    break;
                case GGML_UNARY_OP_STEP:
                    ggml_sycl_step(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU:
                    ggml_sycl_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_sycl_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_sycl_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_ERF:
                    ggml_sycl_gelu_erf(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_sycl_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_sycl_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    ggml_sycl_sigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_sycl_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_sycl_hardswish(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXP:
                    ggml_sycl_exp(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXPM1:
                    ggml_sycl_expm1(ctx, dst);
                    break;
                case GGML_UNARY_OP_SOFTPLUS:
                    ggml_sycl_softplus(ctx, dst);
                    break;
                case GGML_UNARY_OP_SGN:
                    ggml_sycl_sgn(ctx, dst);
                    break;
                case GGML_UNARY_OP_ABS:
                    ggml_sycl_abs(ctx, dst);
                    break;
                case GGML_UNARY_OP_ELU:
                    ggml_sycl_elu(ctx, dst);
                    break;
                case GGML_UNARY_OP_XIELU:
                    ggml_sycl_xielu(ctx, dst);
                    break;
                case GGML_UNARY_OP_FLOOR:
                    ggml_sycl_floor(ctx, dst);
                    break;
                case GGML_UNARY_OP_CEIL:
                    ggml_sycl_ceil(ctx, dst);
                    break;
                case GGML_UNARY_OP_ROUND:
                    ggml_sycl_round(ctx, dst);
                    break;
                case GGML_UNARY_OP_TRUNC:
                    ggml_sycl_trunc(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(dst)) {
                case GGML_GLU_OP_REGLU:
                    ggml_sycl_reglu(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU:
                    ggml_sycl_geglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU:
                    ggml_sycl_swiglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    ggml_sycl_swiglu_oai(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_ERF:
                    ggml_sycl_geglu_erf(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_QUICK:
                    ggml_sycl_geglu_quick(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_sycl_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_sycl_group_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_sycl_op_concat(ctx, dst);
            break;
        case GGML_OP_PAD_REFLECT_1D:
            ggml_sycl_op_pad_reflect_1d(ctx,dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_sycl_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_sycl_pad(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_sycl_leaky_relu(ctx, dst);
            break;
        case GGML_OP_RMS_NORM_BACK:
            ggml_sycl_rms_norm_back(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_sycl_rms_norm(ctx, dst);
            break;
        case GGML_OP_L2_NORM:
            ggml_sycl_l2_norm(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            if (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                return false;
            }
            /* ggml_sycl_mul_mat_id is dependent on ggml_sycl_mul_mat */
            ggml_sycl_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            if (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                return false;
            }
            ggml_sycl_mul_mat_id(ctx, dst);
            break;
        case GGML_OP_OUT_PROD:
            ggml_sycl_op_out_prod(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_sycl_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_sycl_sqr(ctx, dst);
            break;
        case GGML_OP_SQRT:
            ggml_sycl_sqrt(ctx, dst);
            break;
        case GGML_OP_SIN:
            ggml_sycl_sin(ctx, dst);
            break;
        case GGML_OP_COS:
            ggml_sycl_cos(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_sycl_clamp(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_sycl_cpy(ctx, dst->src[0], dst->src[1]);
            break;
        case GGML_OP_CONT:
            ggml_sycl_dup(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            GGML_SYCL_DEBUG("%s: Tensor NO-OP\n", __func__);
            break;
        case GGML_OP_TRI:
            ggml_sycl_op_tri(ctx, dst);
            break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_sycl_diag_mask_inf(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_sycl_op_soft_max(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX_BACK:
            ggml_sycl_op_soft_max_back(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_sycl_cross_entropy_loss(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_sycl_cross_entropy_loss_back(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_sycl_rope(ctx, dst);
            break;
        case GGML_OP_ROPE_BACK:
            ggml_sycl_rope_back(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_sycl_im2col(ctx, dst);
            break;
        case GGML_OP_IM2COL_3D:
            ggml_sycl_im2col_3d(ctx, dst);
            break;
        case GGML_OP_COL2IM_1D:
            ggml_sycl_col2im_1d(ctx, dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_sycl_pool2d(ctx, dst);
            break;
        case GGML_OP_POOL_1D:
            ggml_sycl_pool1d(ctx, dst);
            break;
        case GGML_OP_SUM:
            ggml_sycl_sum(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_sycl_sum_rows(ctx, dst);
            break;
        case GGML_OP_MEAN:
            ggml_sycl_mean(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_sycl_argsort(ctx, dst);
            break;
        case GGML_OP_TOP_K:
            ggml_sycl_op_top_k(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_sycl_op_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV6:
            ggml_sycl_op_rwkv_wkv6(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV7:
            ggml_sycl_op_rwkv_wkv7(ctx, dst);
            break;
        case GGML_OP_GATED_LINEAR_ATTN:
            ggml_sycl_op_gated_linear_attn(ctx, dst);
            break;
        case GGML_OP_GATED_DELTA_NET:
            ggml_sycl_gated_delta_net(ctx, dst);
            break;
        case GGML_OP_GATED_DELTA_NET_REPLAY:
            ggml_sycl_op_gated_delta_net_replay(ctx, dst, nullptr);
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            ggml_sycl_opt_step_adamw(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_SGD:
            ggml_sycl_opt_step_sgd(ctx, dst);
            break;
        case GGML_OP_SSM_CONV:
            ggml_sycl_ssm_conv(ctx, dst);
            break;
        case GGML_OP_SSM_SCAN:
            ggml_sycl_ssm_scan(ctx, dst);
            break;
        case GGML_OP_FILL:
            ggml_sycl_fill(ctx, dst);
            break;
        case GGML_OP_CUMSUM:
            ggml_sycl_cumsum(ctx, dst);
            break;
        case GGML_OP_DIAG:
            ggml_sycl_diag(ctx, dst);
            break;
        case GGML_OP_SOLVE_TRI:
            ggml_sycl_solve_tri(ctx, dst);
            break;
        case GGML_OP_ROLL:
            ggml_sycl_roll(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_sycl_arange(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_sycl_flash_attn_ext(ctx, dst);
            break;
        default:
            return false;
    }

    return true;
} catch (sycl::exception & e) {
    std::cerr << e.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::cerr << "Error OP "<<ggml_op_name(dst->op)<< std::endl;
    std::exit(1);
}

GGML_API void ggml_backend_sycl_get_device_description(int device, char *description,
                                      size_t description_size) try {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_description\n");
    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
        prop, dpct::dev_mgr::instance().get_device(device))));
    snprintf(description, description_size, "%s", prop.get_name());
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void ggml_backend_sycl_get_device_memory(int device, size_t *free,
                                                   size_t *total) try {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_memory\n");
    ggml_sycl_set_device(device);

    SYCL_CHECK(CHECK_TRY_ERROR(
        dpct::dev_mgr::instance().get_device(device).get_memory_info(*free, *total)));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

////////////////////////////////////////////////////////////////////////////////

// backend

static const char * ggml_backend_sycl_get_name(ggml_backend_t backend) {

    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;

    return sycl_ctx->name.c_str();
}

static void ggml_backend_sycl_free(ggml_backend_t backend) {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;

    delete sycl_ctx;
    delete backend;
}

// ARC-LAB: pinned upload ring for ggml_backend_sycl_set_tensor_async (GGML_SYCL_PINNED_UPLOAD)
static char *       g_upload_ring       = nullptr;
static const size_t g_upload_ring_bytes = 32u << 20;
static size_t       g_upload_off        = 0;
// GGML_SYCL_HOST_TIMING=1 also prints the uploads of the first decode-sized graphs (name, bytes)
static void ggml_sycl_log_upload(const ggml_tensor * t, size_t size) {
    static const bool on = getenv("GGML_SYCL_HOST_TIMING") && atoi(getenv("GGML_SYCL_HOST_TIMING"));
    static int        n  = 0;
    if (on && n < 60) {
        ++n;
        fprintf(stderr, "upload: %-28s %zu bytes\n", t->name, size);
    }
}
static void ggml_backend_sycl_set_tensor_async(ggml_backend_t backend,
                                               ggml_tensor *tensor,
                                               const void *data, size_t offset,
                                               size_t size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) && "unsupported buffer type");
    const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    // ARC-LAB: GGML_SYCL_PINNED_UPLOAD=1 stages small per-token inputs (token embedding, positions, masks, state ids) in a
    // pinned host ring first: a copy straight from pageable memory goes through the driver's staging path. The ring
    // restarts at each synchronize (the in-order queue has consumed every staged copy by then).
    if (ggml_sycl_knob("GGML_SYCL_PINNED_UPLOAD", 0) && size <= g_upload_ring_bytes / 4) {
        if (!g_upload_ring) {
            g_upload_ring = (char *) sycl::malloc_host(g_upload_ring_bytes, *stream);
        }
        if (g_upload_ring) {
            if (g_upload_off + size > g_upload_ring_bytes) {
                stream->wait();
                g_upload_off = 0;
            }
            char * staged = g_upload_ring + g_upload_off;
            memcpy(staged, data, size);
            g_upload_off = (g_upload_off + size + 255) & ~(size_t) 255;
            SYCL_CHECK(CHECK_TRY_ERROR((stream)->memcpy((char *) tensor->data + offset, staged, size)));
            ggml_sycl_log_upload(tensor, size);
            return;
        }
    }
    ggml_sycl_log_upload(tensor, size);
    SYCL_CHECK(CHECK_TRY_ERROR(
        (stream)->memcpy((char *)tensor->data + offset, data, size)));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_get_tensor_async(ggml_backend_t backend,
                                               const ggml_tensor *tensor,
                                               void *data, size_t offset,
                                               size_t size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) && "unsupported buffer type");
    const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->memcpy(
        data, (const char *)tensor->data + offset, size)));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static bool ggml_backend_sycl_cpy_tensor_async(ggml_backend_t backend,
                                               const ggml_tensor *src,
                                               ggml_tensor *dst) try {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    bool is_cpy_supported                = dst->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) &&
                            ggml_backend_buffer_is_sycl(src->buffer);
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": dst", dst).c_str());
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" src", src).c_str());
    GGML_SYCL_DEBUG(" is_cpy_supported=%d\n", is_cpy_supported);
    if (is_cpy_supported) {
        /*
        DPCT1009:215: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
        SYCL_CHECK(CHECK_TRY_ERROR((stream)->memcpy(
            dst->data, src->data, ggml_nbytes(dst))));
        return true;
    }

    return false;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// ARC-LAB: GGML_SYCL_HOST_TIMING=1 -> host-side token timing without a tracer (tracers slow the host 2x and invent gaps).
// Per graph: between = host time since the previous synchronize returned (sampling, graph build, logits copy),
// enqueue = time to submit every kernel, wait = synchronize after the last submit. Plus host enqueue time per op type.
// Printed every 64 graphs with more than 500 nodes (decode graphs), then reset.
struct sycl_host_timing {
    bool on = getenv("GGML_SYCL_HOST_TIMING") && atoi(getenv("GGML_SYCL_HOST_TIMING"));
    using clk = std::chrono::steady_clock;
    clk::time_point last_sync_end{}, t_in{}, t_enq{};
    bool   in_big = false;
    int    graphs = 0;
    double between = 0, enqueue = 0, wait = 0, waits_mid = 0;
    long   nodes = 0, syncs = 0;
    std::map<std::string, std::pair<double, long>> per_op;
    void report() {
        if (graphs == 0) return;
        fprintf(stderr, "host-timing: %d graphs, per graph: between %.3f ms, enqueue %.3f ms (%ld nodes), wait %.3f ms, other syncs %.1f\n",
                graphs, between / graphs, enqueue / graphs, nodes / graphs, wait / graphs, (double) syncs / graphs);
        fprintf(stderr, "host-timing: set_tensor (sync uploads) per graph: %.3f ms, %.1f calls, %.0f bytes\n", g_st_ms / graphs,
                (double) g_st_calls / graphs, (double) g_st_bytes / graphs);
        g_st_ms = 0; g_st_calls = 0; g_st_bytes = 0;
        std::vector<std::pair<double, std::string>> v;
        for (auto & kv : per_op) v.push_back({ kv.second.first, kv.first });
        std::sort(v.rbegin(), v.rend());
        for (size_t i = 0; i < v.size() && i < 12; ++i) {
            fprintf(stderr, "host-timing:   %-22s %.3f ms/graph (%ld calls/graph)\n", v[i].second.c_str(), v[i].first / graphs,
                    per_op[v[i].second].second / graphs);
        }
        graphs = 0; between = enqueue = wait = 0; nodes = 0; syncs = 0; per_op.clear();
    }
};
static sycl_host_timing g_ht;

// ARC-LAB: GGML_SYCL_OP_TIMING=1 -> GPU time per op type for draft-sized graphs (fewer than 2000 nodes, every mat-mul at
// most 8 rows: the MTP draft decodes and chains, not the target's verify graph). The queue is drained after every kernel
// group, so each entry is its kernel time plus one launch latency (a serialised upper bound); the sum is reported next to
// the kernel-group count. Printed every 64 such graphs. Timing-only: it slows the draft, never use it for speed runs.
struct sycl_op_timing {
    bool on = getenv("GGML_SYCL_OP_TIMING") && atoi(getenv("GGML_SYCL_OP_TIMING"));
    using clk = std::chrono::steady_clock;
    int    graphs = 0;
    long   groups = 0, nodes = 0;
    double total = 0;
    std::map<std::string, std::pair<double, long>> per_op;
    static bool draft_sized(const ggml_cgraph * g) {
        if (g->n_nodes >= 2000) return false;
        bool mm = false;
        for (int i = 0; i < g->n_nodes; ++i) {
            const ggml_tensor * n = g->nodes[i];
            // weight mat-muls only: a Hadamard transform reshapes each row into 1024-wide blocks (5..17 per row)
            if (n->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(n, 1) != GGML_HINT_SRC0_IS_HADAMARD) {
                mm = true;
                if (n->src[1]->ne[1] * n->src[1]->ne[2] > 8) return false;
            }
        }
        if (mm) {
            static bool logged = false;
            if (!logged) {
                logged = true;
                fprintf(stderr, "op-timing: active (first draft-sized graph: %d nodes)\n", g->n_nodes);
            }
        }
        return mm;
    }
    // op label: mat-muls by weight type (and the Hadamard transforms apart)
    static std::string label(const ggml_tensor * n) {
        if (n->op == GGML_OP_MUL_MAT) {
            if (ggml_get_op_params_i32(n, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                return "MUL_MAT hadamard";
            }
            return std::string("MUL_MAT ") + ggml_type_name(n->src[0]->type);
        }
        return ggml_op_desc(n);
    }
    void report() {
        fprintf(stderr, "op-timing: %d draft-sized graphs, per graph: %.3f ms serialised over %.1f kernel groups (%.0f nodes)\n",
                graphs, total / graphs, (double) groups / graphs, (double) nodes / graphs);
        std::vector<std::pair<double, std::string>> v;
        for (auto & kv : per_op) v.push_back({ kv.second.first, kv.first });
        std::sort(v.rbegin(), v.rend());
        for (size_t i = 0; i < v.size() && i < 16; ++i) {
            const auto & a = per_op[v[i].second];
            fprintf(stderr, "op-timing:   %-22s %.3f ms/graph (%.1f groups/graph, %.1f us each)\n", v[i].second.c_str(),
                    a.first / graphs, (double) a.second / graphs, 1000.0 * a.first / std::max(1L, a.second));
        }
        graphs = 0; groups = 0; nodes = 0; total = 0; per_op.clear();
    }
};
static sycl_op_timing g_ot;

static void ggml_backend_sycl_synchronize(ggml_backend_t backend) try {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    const auto      ht_t0  = sycl_host_timing::clk::now();
    if (g_sync_trace.on) {
        g_sync_trace.note_sync();
    }
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->wait()));
    g_upload_off = 0;
    if (g_ht.on) {
        const auto t1 = sycl_host_timing::clk::now();
        if (g_ht.in_big) {  // first synchronize after a decode graph
            g_ht.wait += std::chrono::duration<double, std::milli>(t1 - ht_t0).count();
            g_ht.in_big = false;
            if (++g_ht.graphs == 64) {
                g_ht.report();
            }
        } else {
            g_ht.syncs++;
        }
        g_ht.last_sync_end = t1;
    }

    GGML_UNUSED(backend);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static bool ggml_sycl_is_view_or_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.
// returns the number of following nodes to skip (0 = no fusion)
// ported from ggml_cuda_try_gdn_cache_fusion - pure graph inspection, backend-agnostic
static int ggml_sycl_try_gdn_cache_fusion(const ggml_cgraph * cgraph, int node_idx,
                                          ggml_sycl_gated_delta_net_fused_cache & fused_state_cpy) {
    if (!g_ggml_sycl_enable_fusion) {
        return 0;
    }

    const ggml_tensor * gdn = cgraph->nodes[node_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output, and the cpy
    // found below is taken to be its only reader, as it is in every graph that builds this op
    if (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    const ggml_tensor * src_v     = gdn->src[2];
    const int64_t       S_v       = src_v->ne[0];
    const int64_t       H         = src_v->ne[1];
    const int64_t       n_tokens  = src_v->ne[2];
    const int64_t       n_seqs    = src_v->ne[3];
    const int64_t       D         = S_v * S_v * H;
    const int64_t       K         = ggml_get_op_params_i32(gdn, 0); // snapshot slot count
    const int64_t       n_written = std::min<int64_t>(n_tokens, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    const size_t tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // the cpy must be the first node the compute loop below runs, so nothing can read the cache first.
    // skip exactly what that loop skips: views, no-ops, and nodes the graph does not compute.
    const ggml_tensor * cpy  = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_sycl_is_view_or_noop(n) || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    const ggml_tensor * src = cpy->src[0]; // view of the gdn snapshot tail
    const ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        return 0;
    }

    // dst is the [D, n_seqs, n_written] cache view, with the per-seq stride D that the kernel assumes.
    // ggml_cpy pins src to the same element count, so src needs no shape check of its own.
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) ||
        dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        return 0;
    }

    fused_state_cpy.data        = (float *) dst->data; // rollback group 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / sizeof(float)) : 0;
    return skip;
}

// ARC-LAB: match GATED_DELTA_NET_REPLAY + the two nodes that store its output in the recurrent cache - a CPY of the
// base state and a SET_ROWS of the update records into the bank picked by its row index - so the kernel writes both
// straight into the cache and the two copies are skipped. Single sequence only. Returns the nodes to skip (0 = none).
static int ggml_sycl_try_gdn_replay_fusion(const ggml_cgraph * cgraph, int node_idx, ggml_sycl_gdn_replay_fused & fused) {
    if (!g_ggml_sycl_enable_fusion) {
        return 0;
    }
    const ggml_tensor * gdn = cgraph->nodes[node_idx];
    if (gdn->op != GGML_OP_GATED_DELTA_NET_REPLAY || gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }
    const ggml_tensor * src_v    = gdn->src[2];
    const int64_t       S_v      = src_v->ne[0];
    const int64_t       H        = src_v->ne[1];
    const int64_t       n_tokens = src_v->ne[2];
    const int64_t       n_seqs   = src_v->ne[3];
    const int64_t       K        = ggml_get_op_params_i32(gdn, 0);
    const int64_t       D        = S_v * S_v * H;
    const int64_t       R        = H * (2 * S_v + 1);
    if (n_seqs != 1) {
        return 0;
    }
    const size_t base_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);
    const size_t rec_off  = base_off + ggml_row_size(GGML_TYPE_F32, D * n_seqs);

    // the next two computed nodes must be the CPY and the SET_ROWS (either order), nothing in between
    const ggml_tensor * cpy  = nullptr;
    const ggml_tensor * sr   = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && (cpy == nullptr || sr == nullptr); ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_sycl_is_view_or_noop(n) || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        if ((n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        if (n->op == GGML_OP_CPY && cpy == nullptr) {
            cpy = n;
        } else if (n->op == GGML_OP_SET_ROWS && sr == nullptr) {
            sr = n;
        } else {
            return 0;
        }
        skip = j - node_idx;
    }
    if (cpy == nullptr || sr == nullptr) {
        return 0;
    }

    // CPY: base region of this op's output -> a contiguous F32 cache view of D * n_seqs floats
    const ggml_tensor * cs = cpy->src[0];
    const ggml_tensor * cd = cpy->src[1];
    if (cs->op != GGML_OP_VIEW || cs->view_src != gdn || cs->view_offs != base_off || !ggml_is_contiguous(cs) ||
        ggml_nelements(cs) != D * n_seqs || cd->type != GGML_TYPE_F32 || cd->data == nullptr ||
        !ggml_is_contiguous(cd)) {
        return 0;
    }

    // SET_ROWS: one row of R * (K - 1) records from the record region -> F32 rows (banks), I64 index on the device
    const ggml_tensor * rs = sr->src[0];
    const ggml_tensor * ri = sr->src[1];
    if (rs->op != GGML_OP_VIEW || rs->view_src != gdn || rs->view_offs != rec_off || !ggml_is_contiguous(rs) ||
        rs->ne[0] != R * (K - 1) || ggml_nrows(rs) != 1 || sr->type != GGML_TYPE_F32 || sr->data == nullptr ||
        sr->ne[0] != R * (K - 1) || sr->nb[0] != sizeof(float) || sr->nb[1] % sizeof(float) != 0 ||
        ri->type != GGML_TYPE_I64 || ggml_nelements(ri) != 1 || ri->data == nullptr ||
        (ri->buffer && ggml_backend_buffer_is_host(ri->buffer))) {
        return 0;
    }

    fused.base           = (float *) cd->data;
    fused.rec            = (float *) sr->data;
    fused.rec_rows       = (const int64_t *) ri->data;
    fused.rec_row_stride = (int64_t) (sr->nb[1] / sizeof(float));
    return skip;
}

// GET_ROWS that gathers a recurrent state row feeding only GATED_DELTA_NET src[5] (possibly through a RESHAPE):
// skip it and let the kernel read the cache row directly. Single sequence only. GGML_SYCL_GDN_GATHER=0 disables.
static bool ggml_sycl_try_gdn_gather_skip(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    static const bool enabled = ggml_sycl_get_env("GGML_SYCL_GDN_GATHER", 1) != 0;
    if (!enabled) {
        return false;
    }
    const ggml_tensor * gr = cgraph->nodes[node_idx];
    if (gr->op != GGML_OP_GET_ROWS || gr->type != GGML_TYPE_F32 || (gr->flags & GGML_TENSOR_FLAG_OUTPUT) ||
        !ggml_is_contiguous(gr)) {
        return false;
    }
    const ggml_tensor * cache = gr->src[0];
    const ggml_tensor * ids   = gr->src[1];
    if (cache->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 || !cache->data || !ids->data ||
        (ids->buffer && ggml_backend_buffer_is_host(ids->buffer)) || cache->nb[0] != sizeof(float) ||
        cache->nb[1] % sizeof(float) != 0 || !ggml_is_contiguous(ids) || ggml_nelements(ids) != 1 ||
        gr->ne[1] != 1 || gr->ne[2] != 1 || gr->ne[3] != 1 || gr->ne[0] != cache->ne[0]) {
        return false;
    }
    if (ggml_node_get_use_count(cgraph, node_idx) != 1) {
        return false;
    }
    const ggml_tensor * cur = gr;
    for (int j = node_idx + 1; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if ((n->op == GGML_OP_GATED_DELTA_NET || n->op == GGML_OP_GATED_DELTA_NET_REPLAY) && n->src[5] == cur) {
            const ggml_tensor * v = n->src[2];
            const int64_t       D = v->ne[0] * v->ne[0] * v->ne[1];
            if (gr->ne[0] != D || v->ne[3] != 1 || ggml_nelements(cur) != D) {
                return false;
            }
            ggml_sycl_gdn_gather_set(ctx, n, (const float *) cache->data, (const int32_t *) ids->data,
                                     (int64_t) (cache->nb[1] / sizeof(float)));
            return true;
        }
        if (n->op == GGML_OP_RESHAPE && n->src[0] == cur) {
            if (ggml_node_get_use_count(cgraph, j) != 1) {
                return false;
            }
            cur = n;
            continue;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (n->src[s] == cur || (n->view_src != nullptr && n->view_src == gr)) {
                return false;
            }
        }
    }
    return false;
}

// ARC-LAB: single-column PTQ1_0 mat-vecs that read the same activation (GDN qkv + z, attention q + k + v) in ONE launch
// with ONE quantization (in-model each launch costs ~10 us beyond its weight bytes). The graph interleaves them with
// other nodes, so later members are pulled forward to node i: only when their output overlaps no tensor that a node in
// between reads or writes (the allocator reuses dead memory). GGML_SYCL_PTQ1_MULTI=0 turns it off.
static bool ggml_sycl_ptq1_multi_eligible(ggml_backend_sycl_context & ctx, const ggml_tensor * mm) {
    if (mm->op != GGML_OP_MUL_MAT || (mm->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        return false;
    }
    const ggml_tensor * w   = mm->src[0];
    const ggml_tensor * act = mm->src[1];
    if (w->type != GGML_TYPE_PTQ1_0 || act->type != GGML_TYPE_F32 || mm->type != GGML_TYPE_F32) {
        return false;
    }
    // 1 column: pairs kernel (rows % 4); 2..8 (MTP verify): decode-once kernel (rows % 2)
    if (act->ne[1] < 1 || act->ne[1] > 8 || act->ne[2] != 1 || act->ne[3] != 1 || w->ne[2] != 1 || w->ne[3] != 1 ||
        w->ne[0] % QK_PTQ1_0 != 0 || w->ne[1] % 4 != 0 || mm->ne[0] != w->ne[1]) {
        return false;
    }
    if (act->ne[1] > 1 && !ggml_sycl_ptq1_0_ncols_dec_enabled()) {
        return false;
    }
    static const bool ncols_on = !getenv("GGML_SYCL_PTQ1_MULTI_NCOLS") || atoi(getenv("GGML_SYCL_PTQ1_MULTI_NCOLS")) != 0;
    if (act->ne[1] > 1 && !ncols_on) {
        return false;
    }
    if (!ggml_is_contiguous(w) || !ggml_is_contiguous(act) || !ggml_is_contiguous(mm)) {
        return false;
    }
    if (ggml_backend_buffer_is_sycl_split(w->buffer) || g_ggml_sycl_prioritize_dmmv || !should_reorder_tensor(ctx, mm)) {
        return false;
    }
    const auto * extra = static_cast<const ggml_tensor_extra_gpu *>(w->extra);
    return extra && extra->optimized_feature.reorder;  // reordered on its first unfused run
}

static bool ggml_sycl_tensors_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    if (!a || !b || !a->data || !b->data) {
        return false;
    }
    const char * a0 = (const char *) a->data;
    const char * b0 = (const char *) b->data;
    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}

// A member whose output memory is still in use by nodes in between (GDN z: the allocator hands its buffer to the conv /
// delta-net intermediates) is computed into a private scratch slot instead and copied to its tensor at its own node.
static constexpr int    PTQ1_MULTI_SLOTS      = 4;
static constexpr size_t PTQ1_MULTI_SLOT_BYTES = 256 * 1024;

static float * ggml_sycl_ptq1_multi_slot(queue_ptr stream) {
    static std::map<int, float *> scratch;  // per device
    static int                    next = 0;
    const int                     dev  = ggml_sycl_get_device();
    float *&                      buf  = scratch[dev];
    if (!buf) {
        buf = (float *) sycl::malloc_device(PTQ1_MULTI_SLOTS * PTQ1_MULTI_SLOT_BYTES, *stream);
        if (!buf) {
            return nullptr;
        }
    }
    next = (next + 1) % PTQ1_MULTI_SLOTS;
    return buf + next * (PTQ1_MULTI_SLOT_BYTES / sizeof(float));
}

static bool ggml_sycl_try_ptq1_multi(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i,
                                     std::vector<uint8_t> & done, std::vector<const void *> & pending) {
    static const bool on  = !getenv("GGML_SYCL_PTQ1_MULTI") || atoi(getenv("GGML_SYCL_PTQ1_MULTI")) != 0;
    static int        log = getenv("GGML_SYCL_PTQ1_MULTI_LOG") ? atoi(getenv("GGML_SYCL_PTQ1_MULTI_LOG")) : 0;
    ggml_tensor * mm0 = cgraph->nodes[i];
    if (!on || !ggml_sycl_ptq1_multi_eligible(ctx, mm0)) {
        return false;
    }
    const ggml_tensor * act = mm0->src[1];
    int  members[3] = { i, -1, -1 };
    bool staged[3]  = { false, false, false };
    int  nm         = 1;
    const int jend = std::min(cgraph->n_nodes, i + 512);
    for (int j = i + 1; j < jend && nm < 3; ++j) {
        ggml_tensor * mj = cgraph->nodes[j];
        if (done[j] || mj->op != GGML_OP_MUL_MAT || mj->src[1] != act || mj->src[0]->ne[0] != mm0->src[0]->ne[0] ||
            !ggml_sycl_ptq1_multi_eligible(ctx, mj)) {
            continue;
        }
        bool safe = true;
        for (int k = i + 1; k < j && safe; ++k) {
            const ggml_tensor * nk = cgraph->nodes[k];
            bool member = false;
            for (int m = 1; m < nm; ++m) {
                member = member || members[m] == k;
            }
            if (member) {
                continue;
            }
            safe = !ggml_sycl_tensors_overlap(mj, nk);
            for (int s = 0; s < GGML_MAX_SRC && safe; ++s) {
                safe = !ggml_sycl_tensors_overlap(mj, nk->src[s]);
            }
        }
        for (int m = 0; m < nm && safe; ++m) {
            safe = !ggml_sycl_tensors_overlap(mj, cgraph->nodes[members[m]]) && !ggml_sycl_tensors_overlap(mj, act);
        }
        if (safe || ggml_nbytes(mj) <= PTQ1_MULTI_SLOT_BYTES) {
            staged[nm]    = !safe;
            members[nm++] = j;
        }
    }
    if (nm < 2) {
        return false;
    }

    const int64_t ne00             = mm0->src[0]->ne[0];
    const int     ne11             = (int) act->ne[1];
    const int     src1_padded_cols = GGML_PAD((int) ne00, MATRIX_ROW_PADDING);
    const int     stride_y_bytes   = src1_padded_cols * (int) sizeof(block_q8_1) / QK8_1;
    const queue_ptr stream         = ctx.stream();
    ggml_sycl_pool_alloc<char> q8(ctx.pool(), (size_t) ne11 * stride_y_bytes);
    if (ne11 == 1) {
        quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>((const float *) act->data, q8.get(), (int) ne00, 1,
                                                              src1_padded_cols, stream);
    } else {
        quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_ptq1_il>((const float *) act->data, q8.get(), (int) ne00, ne11,
                                                                  src1_padded_cols, stream);
    }
    const void * vx[3];
    float *      dst[3];
    int          nrows[3];
    for (int m = 0; m < nm; ++m) {
        ggml_tensor * t = cgraph->nodes[members[m]];
        vx[m]    = t->src[0]->data;
        dst[m]   = (float *) t->data;
        nrows[m] = (int) t->src[0]->ne[1];
        if (staged[m]) {
            dst[m] = ggml_sycl_ptq1_multi_slot(stream);
            if (!dst[m]) {
                return false;
            }
        }
    }
    const bool launched = ne11 == 1 ?
        ggml_sycl_ptq1_0_mul_mat_vec_multi(vx, dst, nrows, nm, q8.get(), (int) ne00, stream) :
        ggml_sycl_ptq1_0_mul_mat_vec_multi_ncols(vx, dst, nrows, nm, q8.get(), (int) ne00, ne11, stride_y_bytes, stream);
    if (!launched) {
        return false;
    }
    if (log > 0) {
        GGML_LOG_INFO("[ptq1-multi] %s", mm0->name);
        for (int m = 1; m < nm; ++m) {
            GGML_LOG_INFO(" + %s (+%d%s)", cgraph->nodes[members[m]]->name, members[m] - i, staged[m] ? ", staged" : "");
        }
        GGML_LOG_INFO("\n");
        --log;
    }
    for (int m = 1; m < nm; ++m) {
        done[members[m]] = 1;
        if (staged[m]) {
            pending[members[m]] = dst[m];  // copied into the tensor when the loop reaches it
        }
    }
    return true;
}

// ARC-LAB: same-input T2 mat-vecs (q+k+v, GDN qkv+z, FFN gate+up; <= 8 tokens) in one launch (GGML_SYCL_T2_MULTI,
// default on). Grouping and safety as ggml_sycl_try_ptq1_multi: a later member computed early must not overlap any node
// in between, else its result is staged in a slot and copied when the loop reaches it.
static bool ggml_sycl_t2_multi_eligible(ggml_backend_sycl_context & ctx, const ggml_tensor * mm) {
    if (mm->op != GGML_OP_MUL_MAT || (mm->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || !ggml_sycl_t2_tensor(mm->src[0])) {
        return false;
    }
    auto * extra = static_cast<ggml_tensor_extra_gpu *>(mm->src[0]->extra);
    if (extra && !extra->optimized_feature.t2 && mm->src[0]->ne[2] == 1) {  // first use: repack here, as the unmerged path
        const ggml_tensor * w = mm->src[0];
        if (!ggml_sycl_t2_repack(*ctx.stream(), w->data, w->ne[0], w->ne[1], w->type == GGML_TYPE_PQ2_0)) {
            return false;
        }
        extra->optimized_feature.t2 = true;
    }
    const ggml_tensor * act   = mm->src[1];
    const int64_t       M     = act->ne[1] * act->ne[2];
    return extra && extra->optimized_feature.t2 && act->type == GGML_TYPE_F32 && mm->type == GGML_TYPE_F32 &&
           act->nb[0] == sizeof(float) && act->ne[3] == 1 && (act->ne[2] == 1 || act->nb[2] == act->ne[1] * act->nb[1]) &&
           M >= 1 && M <= 8 && ggml_is_contiguous(mm) && mm->ne[0] == mm->src[0]->ne[1];
}

static bool ggml_sycl_try_t2_multi(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i, std::vector<uint8_t> & done,
                                   std::vector<const void *> & pending) {
    static int    log = getenv("GGML_SYCL_T2_MULTI_LOG") ? atoi(getenv("GGML_SYCL_T2_MULTI_LOG")) : 0;
    ggml_tensor * mm0 = cgraph->nodes[i];
    if (!ggml_sycl_knob("GGML_SYCL_T2_MULTI", 1) || !ggml_sycl_t2_multi_eligible(ctx, mm0)) {
        return false;
    }
    const ggml_tensor * act = mm0->src[1];
    int  members[3] = { i, -1, -1 };
    bool staged[3]  = { false, false, false };
    int  nm         = 1;
    const int jend = std::min(cgraph->n_nodes, i + 512);
    for (int j = i + 1; j < jend && nm < 3; ++j) {
        ggml_tensor * mj = cgraph->nodes[j];
        if (done[j] || mj->op != GGML_OP_MUL_MAT || mj->src[1] != act || mj->src[0]->ne[0] != mm0->src[0]->ne[0] ||
            !ggml_sycl_t2_multi_eligible(ctx, mj)) {
            continue;
        }
        bool safe = true;
        for (int k = i + 1; k < j && safe; ++k) {
            const ggml_tensor * nk = cgraph->nodes[k];
            bool member = false;
            for (int m = 1; m < nm; ++m) {
                member = member || members[m] == k;
            }
            if (member) {
                continue;
            }
            safe = !ggml_sycl_tensors_overlap(mj, nk);
            for (int s = 0; s < GGML_MAX_SRC && safe; ++s) {
                safe = !ggml_sycl_tensors_overlap(mj, nk->src[s]);
            }
        }
        for (int m = 0; m < nm && safe; ++m) {
            safe = !ggml_sycl_tensors_overlap(mj, cgraph->nodes[members[m]]) && !ggml_sycl_tensors_overlap(mj, act);
        }
        if (safe || ggml_nbytes(mj) <= PTQ1_MULTI_SLOT_BYTES) {
            staged[nm]    = !safe;
            members[nm++] = j;
        }
    }
    if (nm < 2) {
        return false;
    }
    const queue_ptr stream = ctx.stream();
    const void *    w[3];
    float *         dst[3];
    int64_t         N[3];
    for (int m = 0; m < nm; ++m) {
        ggml_tensor * t = cgraph->nodes[members[m]];
        w[m]   = t->src[0]->data;
        dst[m] = (float *) t->data;
        N[m]   = t->src[0]->ne[1];
        if (staged[m]) {
            dst[m] = ggml_sycl_ptq1_multi_slot(stream);
            if (!dst[m]) {
                return false;
            }
        }
    }
    const void * key = act->view_src && act->view_offs == 0 ? act->view_src : act;
    if (!ggml_sycl_t2_mul_mat_multi(*stream, nm, w, dst, N, (const float *) act->data, act->nb[1] / sizeof(float),
                                    act->ne[1] * act->ne[2], act->ne[0], key)) {
        return false;
    }
    if (log > 0) {
        fprintf(stderr, "[t2-multi] %s", mm0->name);
        for (int m = 1; m < nm; ++m) {
            fprintf(stderr, " + %s (+%d%s)", cgraph->nodes[members[m]]->name, members[m] - i, staged[m] ? ", staged" : "");
        }
        fprintf(stderr, "\n");
        --log;
    }
    for (int m = 1; m < nm; ++m) {
        done[members[m]] = 1;
        if (staged[m]) {
            pending[members[m]] = dst[m];
        }
    }
    return true;
}

// ARC-LAB: see the call in ggml_backend_sycl_graph_compute_impl. Advances i past the nodes it computed.
static bool ggml_sycl_try_add_norm_rot(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int & i,
                                       std::vector<uint8_t> & done) {
    const int     n0  = i;
    ggml_tensor * add = cgraph->nodes[n0]->op == GGML_OP_ADD ? cgraph->nodes[n0] : nullptr;
    const int     in  = add ? n0 + 1 : n0;  // RMS_NORM
    if (in + 1 >= cgraph->n_nodes) {
        return false;
    }
    ggml_tensor * norm = cgraph->nodes[in];
    ggml_tensor * mulw = cgraph->nodes[in + 1];
    if (norm->op != GGML_OP_RMS_NORM || mulw->op != GGML_OP_MUL || mulw->src[0] != norm) {
        return false;
    }
    if (add && norm->src[0] != add) {
        return false;
    }
    const ggml_tensor * x = add ? add : norm->src[0];
    const ggml_tensor * w = mulw->src[1];
    const int64_t       K = x->ne[0], M = ggml_nrows(x);
    auto f32c = [](const ggml_tensor * t) { return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t); };
    if (!f32c(x) || !f32c(norm) || !f32c(mulw) || !f32c(w) || w->ne[0] != K || ggml_nrows(w) != 1 || x->ne[3] != 1) {
        return false;
    }
    if (add && (!f32c(add->src[0]) || !f32c(add->src[1]) || !ggml_are_same_shape(add->src[0], add) ||
                !ggml_are_same_shape(add->src[1], add))) {
        return false;
    }
    // rotation tail: MUL(signs) + RESHAPE + MUL_MAT(hint Hadamard 1024)
    ggml_tensor * mm    = nullptr;
    const float * signs = nullptr;
    if (in + 4 < cgraph->n_nodes) {
        ggml_tensor * ms = cgraph->nodes[in + 2];
        ggml_tensor * rs = cgraph->nodes[in + 3];
        ggml_tensor * mt = cgraph->nodes[in + 4];
        const ggml_tensor * sg = ms->src[1];
        if (ms->op == GGML_OP_MUL && ms->src[0] == mulw && rs->op == GGML_OP_RESHAPE && rs->src[0] == ms &&
            mt->op == GGML_OP_MUL_MAT && mt->src[1] == rs && ggml_get_op_params_i32(mt, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
            mt->src[0]->ne[0] == 1024 && f32c(mt) && f32c(ms) && sg && f32c(sg) && sg->ne[0] == K && ggml_nrows(sg) == 1) {
            mm    = mt;
            signs = (const float *) sg->data;
        }
    }
    // DeltaNet layers: the rotation comes a few nodes later (conv-cache ops in between). Fold it in and write its output
    // here, early, if no node in between touches that memory (GGML_SYCL_NORM_ROT_FAR, default on).
    int ms_far = -1;
    if (!mm && add && ggml_sycl_knob("GGML_SYCL_NORM_ROT_FAR", 1)) {
        for (int j = in + 2; j + 2 < cgraph->n_nodes && j <= in + 32; ++j) {
            ggml_tensor * ms = cgraph->nodes[j];
            if (ms->op != GGML_OP_MUL || ms->src[0] != mulw) {
                continue;
            }
            ggml_tensor *       rs = cgraph->nodes[j + 1];
            ggml_tensor *       mt = cgraph->nodes[j + 2];
            const ggml_tensor * sg = ms->src[1];
            if (!done[j] && !done[j + 1] && !done[j + 2] && rs->op == GGML_OP_RESHAPE && rs->src[0] == ms &&
                mt->op == GGML_OP_MUL_MAT && mt->src[1] == rs && ggml_get_op_params_i32(mt, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
                mt->src[0]->ne[0] == 1024 && f32c(mt) && f32c(ms) && sg && f32c(sg) && sg->ne[0] == K && ggml_nrows(sg) == 1 &&
                sg->op == GGML_OP_NONE) {  // the signs are read early: they must be a weight, not computed in between
                bool clear = true;  // nodes computed between the norm and the rotation must not touch the rotation output
                for (int k = in + 2; k < j && clear; ++k) {
                    const ggml_tensor * nk = cgraph->nodes[k];
                    if (ggml_sycl_is_view_or_noop(nk) || (nk->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                        continue;
                    }
                    clear = !ggml_sycl_tensors_overlap(mt, nk);
                    for (int t = 0; t < GGML_MAX_SRC && clear; ++t) {
                        clear = !ggml_sycl_tensors_overlap(mt, nk->src[t]);
                    }
                }
                if (clear) {
                    ms_far = j;
                    mm     = mt;
                    signs  = (const float *) sg->data;
                }
            }
            break;  // only the first MUL of the norm output is a candidate
        }
    }
    if (!add && !mm) {
        return false;  // plain RMS_NORM + MUL: ggml_sycl_op_rms_norm_fused below
    }
    const int last = mm ? (ms_far >= 0 ? ms_far + 2 : in + 4) : in + 1;
    // nodes inside the chain must have no other consumers; write the ADD and the weighted norm only if used after it
    std::vector<int> idx;
    for (int j = n0; j <= in + 1; ++j) idx.push_back(j);
    if (mm) {
        const int t0 = ms_far >= 0 ? ms_far : in + 2;
        for (int j = t0; j <= t0 + 2; ++j) idx.push_back(j);
    }
    std::vector<int> outs;
    if (add) outs.push_back(n0);
    outs.push_back(in + 1);
    if (mm) outs.push_back(last);
    std::vector<ggml_op> ops;
    for (int j : idx) ops.push_back(cgraph->nodes[j]->op);
    if (!ggml_can_fuse_subgraph_ext(cgraph, idx.data(), (int) idx.size(), ops.data(), outs.data(), (int) outs.size())) {
        return false;
    }
    auto used_after = [&](int j) {
        int inside = 0;
        for (int k : idx) {
            if (k <= j) continue;
            for (int s = 0; s < GGML_MAX_SRC; ++s) inside += cgraph->nodes[k]->src[s] == cgraph->nodes[j];
        }
        return (cgraph->nodes[j]->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_node_get_use_count(cgraph, j) > inside;
    };
    float * add_out = add && used_after(n0) ? (float *) add->data : nullptr;
    float * normed  = !mm || used_after(in + 1) ? (float *) mulw->data : nullptr;
    // across tokens a work-group may write where another still reads: outputs must miss the inputs or match them exactly
    if (M > 1) {
        const ggml_tensor * ins[2]  = { add ? add->src[0] : x, add ? add->src[1] : nullptr };
        std::pair<const void *, size_t> outr[3] = { { add_out, add_out ? ggml_nbytes(add) : 0 },
                                                    { normed, normed ? ggml_nbytes(mulw) : 0 },
                                                    { mm ? mm->data : nullptr, mm ? ggml_nbytes(mm) : 0 } };
        for (const auto & o : outr) {
            for (const ggml_tensor * t : ins) {
                if (!o.first || !t) continue;
                const char * ob = (const char *) o.first, * tb = (const char *) t->data;
                if (ob != tb && ob < tb + ggml_nbytes(t) && tb < ob + o.second) {
                    return false;
                }
            }
        }
    }
    float eps;
    memcpy(&eps, norm->op_params, sizeof(float));
    if (!ggml_sycl_t2_add_norm_rot(*ctx.stream(), add ? (const float *) add->src[0]->data : (const float *) x->data,
                                   add ? (const float *) add->src[1]->data : nullptr, add_out, (const float *) w->data, eps,
                                   normed, signs, mm ? (float *) mm->data : nullptr, M, K, mm)) {
        return false;
    }
    static bool fired_logged = false;
    if (!fired_logged && getenv("GGML_SYCL_FUSION_LOG")) {
        fired_logged = true;
        fprintf(stderr, "ggml_sycl_try_add_norm_rot: fused (%s%s%s)\n", add ? "add + " : "", "norm", mm ? " + rotation" : "");
    }
    if (ms_far >= 0) {
        done[ms_far] = done[ms_far + 1] = done[ms_far + 2] = 1;  // computed here, skipped when the loop gets there
        i = in + 1;
        static bool logged = false;
        if (!logged) {
            logged = true;
            fprintf(stderr, "%s: norm + rotation %d nodes apart fused\n", __func__, ms_far - (in + 1));
        }
    } else {
        i = last;
    }
    return true;
}

// ARC-LAB: DeltaNet conv chain for one token: CONCAT(conv state, new qkv) -> CPY(newest d_conv-1 columns -> conv cache) ->
// SSM_CONV -> SILU -> L2_NORM(q/k heads) in one kernel (one work-group per head of S channels), instead of 5 launches.
// The nodes are spread over ~15 graph nodes with cache ops in between; the fused outputs are written at the CONCAT's
// position, so every node computed in between must not touch their memory (checked), and they must not overlap inputs.
// GGML_SYCL_CONV_CHAIN=0 turns it off.
static bool ggml_sycl_try_conv_chain(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i, std::vector<uint8_t> & done) {
    if (!ggml_sycl_knob("GGML_SYCL_CONV_CHAIN", 1)) {
        return false;
    }
    ggml_tensor * cat = cgraph->nodes[i];
    if (cat->op != GGML_OP_CONCAT || cat->type != GGML_TYPE_F32 || ggml_get_op_params_i32(cat, 0) != 0) {
        return false;
    }
    const int64_t dc = cat->ne[0], C = cat->ne[1];
    const ggml_tensor * cs = cat->src[0];
    const ggml_tensor * x  = cat->src[1];
    if (cat->ne[2] != 1 || cat->ne[3] != 1 || dc < 2 || dc > 8 || !ggml_is_contiguous(cat) || cs->type != GGML_TYPE_F32 ||
        x->type != GGML_TYPE_F32 || cs->ne[0] != dc - 1 || cs->ne[1] != C || !ggml_is_contiguous(cs) || x->ne[0] != 1 ||
        x->ne[1] != C) {
        return false;
    }
    int ic = -1, iv = -1, iconv = -1, isilu = -1, il2 = -1;
    const int jmax = std::min(cgraph->n_nodes - 1, i + 48);
    for (int j = i + 1; j <= jmax && il2 < 0; ++j) {
        ggml_tensor * n = cgraph->nodes[j];
        if (n->op == GGML_OP_CPY && n->src[0]->op == GGML_OP_VIEW && n->src[0]->view_src == cat) {
            ic = j;
        } else if (n->op == GGML_OP_SSM_CONV && n->src[0] == cat) {
            iconv = j;
        } else if (iconv >= 0 && n->op == GGML_OP_UNARY && ggml_get_unary_op(n) == GGML_UNARY_OP_SILU &&
                   n->src[0] == cgraph->nodes[iconv]) {
            isilu = j;
        } else if (isilu >= 0 && n->op == GGML_OP_L2_NORM && n->src[0]->op == GGML_OP_VIEW &&
                   n->src[0]->view_src == cgraph->nodes[isilu]) {
            il2 = j;
            iv  = j;
        }
    }
    static int reasons = 0;
    auto why = [&](const char * r) {
        if (reasons < 3 && getenv("GGML_SYCL_CONV_CHAIN_DEBUG")) {
            ++reasons;
            fprintf(stderr, "conv chain: not fused at node %d (%s): %s\n", i, cat->name, r);
        }
        return false;
    };
    if (ic < 0 || iconv < 0 || isilu < 0 || il2 < 0 || done[ic] || done[iconv] || done[isilu] || done[il2]) {
        return why("chain nodes not found");
    }
    ggml_tensor *       cpy  = cgraph->nodes[ic];
    ggml_tensor *       conv = cgraph->nodes[iconv];
    ggml_tensor *       silu = cgraph->nodes[isilu];
    ggml_tensor *       l2   = cgraph->nodes[il2];
    const ggml_tensor * cv   = cpy->src[0];  // view: columns 1..dc-1 of the concat
    const ggml_tensor * dst  = cpy->src[1];  // conv cache slot
    const ggml_tensor * w    = conv->src[1];
    const ggml_tensor * l2v  = l2->src[0];
    const int64_t       S = l2v->ne[0], H = l2v->ne[1];
    if (cv->ne[0] != dc - 1 || cv->ne[1] != C || cv->view_offs != sizeof(float) || cv->nb[1] != cat->nb[1] ||
        dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) || ggml_nelements(dst) != (dc - 1) * C ||
        w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) || w->ne[0] != dc || w->ne[1] != C ||
        ggml_nelements(conv) != C || !ggml_is_contiguous(conv) || !ggml_is_contiguous(silu) || silu->type != GGML_TYPE_F32 ||
        l2v->view_offs != 0 || l2v->nb[1] != (size_t) S * sizeof(float) || l2v->ne[2] != 1 || l2v->ne[3] != 1 ||
        !ggml_is_contiguous(l2) || S < 16 || S > 1024 || S % 16 != 0 || C % S != 0 || S * H > C ||
        (cpy->flags & GGML_TENSOR_FLAG_OUTPUT) || (conv->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return why("shapes/types");
    }
    // the concat and the raw conv output must have no readers outside the chain
    for (int j = i + 1; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        for (int k = 0; k < GGML_MAX_SRC && n->src[k]; ++k) {
            const ggml_tensor * sj = n->src[k];
            if ((sj == cat && j != iconv && n != cv) || (sj == conv && j != isilu) || (sj->view_src == cat && sj != cv)) {
                return why("outside reader");
            }
        }
        if (n->view_src == cat && n != cv) {
            return why("outside view");
        }
    }
    auto overlap = [](const void * a, size_t na, const void * b, size_t nb) {
        const char * pa = (const char *) a, * pb = (const char *) b;
        return na && nb && pa < pb + nb && pb < pa + na;
    };
    const void * outs[3]  = { silu->data, l2->data, dst->data };
    const size_t outn[3]  = { ggml_nbytes(silu), ggml_nbytes(l2), ggml_nbytes(dst) };
    const ggml_tensor * ins[3] = { cs, x, w };
    // an output may sit exactly on x (the planner reuses its buffer after the concat): x and the silu / l2 outputs keep
    // channel c at element c, a lane reads x[c] before writing c, and l2 is written after the work-group barrier
    static const char * out_names[3] = { "silu", "l2", "cache" };
    static const char * in_names[3]  = { "conv state", "x", "weights" };
    // l2 on top of the conv state (other lanes still read it): leave L2_NORM to its own kernel after this one
    const bool l2_in = !overlap(outs[1], outn[1], cs->data, ggml_nbytes(cs));
    for (int o = 0; o < 3; ++o) {
        if (o == 1 && !l2_in) {
            continue;
        }
        for (int t = 0; t < 3; ++t) {
            if (!overlap(outs[o], outn[o], ins[t]->data, ggml_nbytes(ins[t]))) {
                continue;
            }
            const bool same_slot_ok = t == 1 && o < 2 && outs[o] == ins[t]->data && x->nb[1] == sizeof(float);
            if (!same_slot_ok) {
                static char msg[96];
                snprintf(msg, sizeof(msg), "output %s overlaps input %s", out_names[o], in_names[t]);
                return why(msg);
            }
        }
    }
    // nodes computed between the concat and the last fused node must not touch the early-written outputs
    for (int j = i + 1; j <= il2; ++j) {
        if (j == ic || j == iconv || j == isilu || j == il2) {
            continue;
        }
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_sycl_is_view_or_noop(n) || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        const int prod[3] = { isilu, il2, ic };  // where each output is produced in the original order
        for (int o = 0; o < 3; ++o) {
            if (o == 1 && !l2_in) {
                continue;
            }
            if (j > prod[o] && o == 2) {
                continue;  // the cache slot is final once the original CPY would have run
            }
            if (overlap(outs[o], outn[o], n->data, ggml_nbytes(n))) {
                return why("node in between writes output memory");
            }
            const ggml_tensor * own[3] = { silu, l2, cpy };
            for (int k = 0; k < GGML_MAX_SRC && n->src[k]; ++k) {
                const ggml_tensor * sk = n->src[k];
                if (j > prod[o] && (sk == own[o] || sk->view_src == own[o])) {
                    continue;  // a consumer of this output, after its producer: reads the right value
                }
                if (overlap(outs[o], outn[o], sk->data, ggml_nbytes(sk))) {
                    return why("node in between reads output memory");
                }
            }
        }
    }
    float eps;
    memcpy(&eps, l2->op_params, sizeof(float));
    const float * csd = (const float *) cs->data;
    const char *  xd  = (const char *) x->data;
    const size_t  xnb = x->nb[1];
    const float * wd  = (const float *) w->data;
    float *       dd  = (float *) dst->data;
    float *       sd  = (float *) silu->data;
    float *       ld  = (float *) l2->data;
    const int     D   = (int) dc;
    const int64_t HL  = l2_in ? H : 0;
    ctx.stream()->parallel_for(sycl::nd_range<1>((size_t) C, (size_t) S), [=](sycl::nd_item<1> it) {
        const int64_t c = it.get_global_id(0);
        const int64_t g = it.get_group(0);
        float         v[8];
        for (int t = 0; t < D - 1; ++t) {
            v[t] = csd[c * (D - 1) + t];
        }
        v[D - 1] = *(const float *) (xd + c * xnb);
        float sum = 0.0f;
        for (int t = 0; t < D; ++t) {
            sum += v[t] * wd[c * D + t];
        }
        for (int t = 1; t < D; ++t) {
            dd[c * (D - 1) + t - 1] = v[t];
        }
        const float y = op_silu(sum);
        sd[c]         = y;
        if (g < HL) {  // uniform per work-group
            const float ss    = sycl::reduce_over_group(it.get_group(), y * y, sycl::plus<float>());
            const float scale = sycl::rsqrt(sycl::fmax(ss, eps * eps));
            ld[g * S + it.get_local_id(0)] = scale * y;
        }
    });
    done[ic] = done[iconv] = done[isilu] = 1;
    if (l2_in) {
        done[il2] = 1;
    }
    static bool logged = false;
    if (!logged) {
        logged = true;
        fprintf(stderr, "%s: fused DeltaNet conv chain active (C=%lld, d_conv=%lld, %lld heads of %lld for L2)\n", __func__,
                      (long long) C, (long long) dc, (long long) H, (long long) S);
    }
    GGML_UNUSED(iv);
    return true;
}

// ARC-LAB: DeltaNet gate block in one kernel (GGML_SYCL_DN_GATES, default on):
//   alpha = MUL_MAT(bf16 W_a, x) -> RESHAPE -> ADD(dt bias) -> SOFTPLUS -> MUL(ssm_a) -> RESHAPE
//   beta  = MUL_MAT(bf16 W_b, x) -> RESHAPE -> SIGMOID
// 9 graph nodes / 7 launches per layer for 2 x 48 outputs; here one work-group per row of both matrices, up to 8 tokens.
static bool ggml_sycl_try_dn_gates(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int & i) {
    if (!ggml_sycl_knob("GGML_SYCL_DN_GATES", 1) || i + 8 >= cgraph->n_nodes) {
        return false;
    }
    ggml_tensor * const * n = cgraph->nodes + i;
    static int dbg = getenv("GGML_SYCL_DN_GATES_DEBUG") ? 6 : 0;
    const bool bf16_mm = n[0]->src[0] && n[0]->src[0]->type == GGML_TYPE_BF16;
    auto why = [&](const char * r) {
        if (dbg > 0 && bf16_mm) {
            --dbg;
            fprintf(stderr, "dn gates: not fused at node %d (%s, src1 ne %lld %lld %lld): %s; next ops:", i, n[0]->name,
                    (long long) n[0]->src[1]->ne[0], (long long) n[0]->src[1]->ne[1], (long long) n[0]->src[1]->ne[2], r);
            for (int k = 1; k < 9 && i + k < cgraph->n_nodes; ++k) fprintf(stderr, " %s", ggml_op_desc(n[k]));
            fprintf(stderr, "\n");
        }
        return false;
    };
    const ggml_op ops[9] = { GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_MUL,
                             GGML_OP_RESHAPE, GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_UNARY };
    for (int k = 0; k < 9; ++k) {
        if (n[k]->op != ops[k]) {
            return why("op sequence");
        }
    }
    if (ggml_get_unary_op(n[3]) != GGML_UNARY_OP_SOFTPLUS || ggml_get_unary_op(n[8]) != GGML_UNARY_OP_SIGMOID ||
        n[1]->src[0] != n[0] || n[2]->src[0] != n[1] || n[3]->src[0] != n[2] || n[4]->src[0] != n[3] ||
        n[5]->src[0] != n[4] || n[7]->src[0] != n[6] || n[8]->src[0] != n[7] || n[6]->src[1] != n[0]->src[1]) {
        return why("edges");
    }
    const ggml_tensor * wa = n[0]->src[0];
    const ggml_tensor * wb = n[6]->src[0];
    const ggml_tensor * x  = n[0]->src[1];
    const ggml_tensor * bias = n[2]->src[1];
    const ggml_tensor * sa   = n[4]->src[1];
    const int64_t K = wa->ne[0], R = wa->ne[1], M = x->ne[1] * x->ne[2];
    auto f32c = [](const ggml_tensor * t) { return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t); };
    if (wa->type != GGML_TYPE_BF16 || wb->type != GGML_TYPE_BF16 || !ggml_is_contiguous(wa) || !ggml_is_contiguous(wb) ||
        wb->ne[0] != K || wb->ne[1] != R || !f32c(x) || x->ne[0] != K || x->ne[3] != 1 || M < 1 || M > 8 || K % 8 != 0 ||
        !f32c(bias) || ggml_nelements(bias) != R || !f32c(sa) || ggml_nelements(sa) != R || !f32c(n[4]) || !f32c(n[8]) ||
        ggml_nelements(n[4]) != R * M || ggml_nelements(n[8]) != R * M || !f32c(n[0]) || !f32c(n[6])) {
        return why("shapes/types");
    }
    {
        int idx[9];
        for (int k = 0; k < 9; ++k) idx[k] = i + k;
        const int outs[3] = { i + 4, i + 5, i + 8 };
        if (!ggml_can_fuse_subgraph_ext(cgraph, idx, 9, ops, outs, 3)) {
            return why("outside readers");
        }
    }
    auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        const char * pa = (const char *) a->data, * pb = (const char *) b->data;
        return pa < pb + ggml_nbytes(b) && pb < pa + ggml_nbytes(a);
    };
    // every work-group reads all of x: an output on top of an input (the planner reuses x's buffer once its last
    // reader is done) goes through a scratch buffer and one small copy kernel
    bool staged = false;
    for (const ggml_tensor * o : { n[4], n[8] }) {
        for (const ggml_tensor * in : { x, wa, wb, bias, sa }) {
            staged |= overlap(o, in);
        }
    }
    ggml_sycl_pool_alloc<float> stage(ctx.pool());
    float * sa_out = nullptr;
    if (staged) {
        sa_out = stage.alloc((size_t) 2 * R * M);
    }
    const uint16_t * pa = (const uint16_t *) wa->data;
    const uint16_t * pb = (const uint16_t *) wb->data;
    const float *    px = (const float *) x->data;
    const float *    pbias = (const float *) bias->data;
    const float *    psa   = (const float *) sa->data;
    float *          oa    = staged ? sa_out : (float *) n[4]->data;
    float *          ob    = staged ? sa_out + R * M : (float *) n[8]->data;
    const int        Ki = (int) K, Ri = (int) R, Mi = (int) M;
    constexpr int    WG = 256;
    ctx.stream()->parallel_for(sycl::nd_range<1>((size_t) 2 * R * WG, WG), [=](sycl::nd_item<1> it) {
        const int        g   = it.get_group(0);
        const bool       isb = g >= Ri;
        const int        row = isb ? g - Ri : g;
        const uint16_t * w   = (isb ? pb : pa) + (size_t) row * Ki;
        const int        tid = it.get_local_id(0);
        float            acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
        for (int k = tid * 8; k < Ki; k += WG * 8) {
            const sycl::vec<uint16_t, 8> wv = *(const sycl::vec<uint16_t, 8> *) (w + k);
            float wf[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) wf[j] = sycl::bit_cast<float>((uint32_t) wv[j] << 16);
            for (int m = 0; m < Mi; ++m) {
                const float * xr = px + (size_t) m * Ki + k;
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[m] += wf[j] * xr[j];
            }
        }
        for (int m = 0; m < Mi; ++m) {
            const float sum = sycl::reduce_over_group(it.get_group(), acc[m], sycl::plus<float>());
            if (tid == 0) {
                if (isb) {
                    ob[m * Ri + row] = 1.0f / (1.0f + sycl::exp(-sum));
                } else {
                    const float v  = sum + pbias[row];
                    const float sp = sycl::fmax(v, 0.0f) + sycl::log1p(sycl::exp(-sycl::fabs(v)));
                    oa[m * Ri + row] = sp * psa[row];
                }
            }
        }
    });
    if (staged) {
        float * da = (float *) n[4]->data, * db = (float *) n[8]->data;
        const float * so = sa_out;
        const size_t  RM = (size_t) R * M;
        ctx.stream()->parallel_for(sycl::range<1>(2 * RM), [=](sycl::id<1> j) {
            if (j < RM) {
                da[j] = so[j];
            } else {
                db[j - RM] = so[j];
            }
        });
    }
    static bool logged = false, logged_staged = false;
    if (staged && !logged_staged) {
        logged_staged = true;
        fprintf(stderr, "%s: fused DeltaNet gates active via scratch (%lld tokens)\n", __func__, (long long) M);
    }
    if (!logged) {
        logged = true;
        fprintf(stderr, "%s: fused DeltaNet gates active (K=%lld, 2 x %lld rows, %lld tokens)\n", __func__, (long long) K,
                (long long) R, (long long) M);
    }
    i += 8;
    return true;
}

// ARC-LAB: a kernel whose work-group reads block r of an input and later writes block r of its output may run in place
// (same base address), never with a partial overlap
static bool ggml_sycl_same_or_disjoint(const ggml_tensor * out, const ggml_tensor * in) {
    const char * po = (const char *) out->data, * pi = (const char *) in->data;
    return po == pi || po + ggml_nbytes(out) <= pi || pi + ggml_nbytes(in) <= po;
}

static void ggml_backend_sycl_graph_compute_impl(ggml_backend_sycl_context * sycl_ctx, ggml_cgraph * cgraph) {
    ggml_sycl_set_main_device(sycl_ctx->device);
    ggml_sycl_gdn_gather_reset(*sycl_ctx);
    // ARC-LAB: nodes already computed by ggml_sycl_try_ptq1_multi, and staged outputs still to copy in
    std::vector<uint8_t>      done(cgraph->n_nodes, 0);
    std::vector<const void *> pending(cgraph->n_nodes, nullptr);
    const bool                       ht_on   = g_ht.on && cgraph->n_nodes > 500;
    sycl_host_timing::clk::time_point ht_prev = sycl_host_timing::clk::now();
    const char *                     ht_op   = nullptr;
    const bool                       ot_on   = g_ot.on && sycl_op_timing::draft_sized(cgraph);
    sycl_op_timing::clk::time_point  ot_prev{};
    std::string                      ot_op;
    auto ot_mark = [&](const std::string & next) {  // drain the queue, charge the time since the last mark to the op that ran
        SYCL_CHECK(CHECK_TRY_ERROR(sycl_ctx->stream()->wait()));
        const auto now = sycl_op_timing::clk::now();
        if (!ot_op.empty()) {
            const double ms = std::chrono::duration<double, std::milli>(now - ot_prev).count();
            auto & a = g_ot.per_op[ot_op];
            a.first += ms;
            a.second++;
            g_ot.total += ms;
            g_ot.groups++;
        }
        ot_prev = now;
        ot_op   = next;
    };
    if (g_sync_trace.on) {
        g_sync_trace.note_graph(cgraph);
    }
    if (ot_on) {
        ot_mark(std::string());  // inputs and earlier work are not charged
        g_ot.nodes += cgraph->n_nodes;
    }
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (ht_on) {
            const auto now = sycl_host_timing::clk::now();
            if (ht_op) {
                auto & a = g_ht.per_op[ht_op];
                a.first += std::chrono::duration<double, std::milli>(now - ht_prev).count();
                a.second++;
            }
            ht_prev = now;
            ht_op   = ggml_op_desc(node);
        }
        if (done[i]) {
            if (pending[i]) {
                SYCL_CHECK(CHECK_TRY_ERROR(sycl_ctx->stream()->memcpy(node->data, pending[i], ggml_nbytes(node))));
            }
            continue;
        }
        if (ggml_sycl_is_view_or_noop(node)) {
            continue;
        }
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        if (ot_on) {
            ot_mark(sycl_op_timing::label(node));
        }

        const int nodes_to_skip = ggml_sycl_fuse(*sycl_ctx, cgraph, i);
        if (nodes_to_skip != 0) {
            i += nodes_to_skip;
            continue;
        }
#ifndef NDEBUG
        assert(node->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device));
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (node->src[j] != nullptr) {
                assert(node->src[j]->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device));
            }
        }
#endif
        if (node->op == GGML_OP_GET_ROWS && ggml_sycl_try_gdn_gather_skip(*sycl_ctx, cgraph, i)) {
            continue;
        }
        // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
        if (node->op == GGML_OP_GATED_DELTA_NET) {
            ggml_sycl_gated_delta_net_fused_cache fused_state_cpy;
            const int gdn_nodes_to_skip = ggml_sycl_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
            if (gdn_nodes_to_skip > 0) {
                ggml_sycl_op_gated_delta_net_fused_cache(*sycl_ctx, node, fused_state_cpy);
                i += gdn_nodes_to_skip;
                continue;
            }
        }
        if (node->op == GGML_OP_GATED_DELTA_NET_REPLAY) {
            ggml_sycl_gdn_replay_fused fused;
            const int rpl_nodes_to_skip = ggml_sycl_try_gdn_replay_fusion(cgraph, i, fused);
            if (rpl_nodes_to_skip > 0) {
                ggml_sycl_op_gated_delta_net_replay(*sycl_ctx, node, &fused);
                i += rpl_nodes_to_skip;
                continue;
            }
        }
        if (node->op == GGML_OP_MUL_MAT && ggml_sycl_try_dn_gates(*sycl_ctx, cgraph, i)) {
            continue;
        }
        if (node->op == GGML_OP_CONCAT && ggml_sycl_try_conv_chain(*sycl_ctx, cgraph, i, done)) {
            continue;
        }
        // ARC-LAB: [residual ADD +] RMS_NORM + MUL(weight) [+ MUL(signs) + RESHAPE + FWHT-hint MUL_MAT] in one kernel
        // (ggml_sycl_t2_add_norm_rot): the post-attention norm of every layer and the pre-attention norm of the full
        // attention layers end in the rotation that feeds the T2 mat-vecs; the DeltaNet layers' norm is followed by
        // cache ops first, so there only the ADD folds in.
        if ((node->op == GGML_OP_ADD || node->op == GGML_OP_RMS_NORM) && ggml_sycl_try_add_norm_rot(*sycl_ctx, cgraph, i, done)) {
            continue;
        }
        // ARC-LAB: FFN SWIGLU -> MUL(signs) -> RESHAPE -> FWHT-hint MUL_MAT: the rotation computes silu(gate) * up on its
        // load (GGML_SYCL_GLU_ROT, default on)
        if (node->op == GGML_OP_GLU && i + 3 < cgraph->n_nodes && ggml_sycl_knob("GGML_SYCL_GLU_ROT", 1)) {
            ggml_tensor * ms = cgraph->nodes[i + 1];
            ggml_tensor * rs = cgraph->nodes[i + 2];
            ggml_tensor * mt = cgraph->nodes[i + 3];
            const ggml_tensor * g  = node->src[0];
            const ggml_tensor * u  = node->src[1];
            const ggml_tensor * sg = ms->src[1];
            const ggml_op       gops[4] = { GGML_OP_GLU, GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_MUL_MAT };
            const int           gidx[4] = { i, i + 1, i + 2, i + 3 };
            const int           gout[1] = { i + 3 };
            if (ggml_get_glu_op(node) == GGML_GLU_OP_SWIGLU && u && ggml_get_op_params_i32(node, 1) == 0 &&
                ms->op == GGML_OP_MUL && ms->src[0] == node && rs->op == GGML_OP_RESHAPE && rs->src[0] == ms &&
                mt->op == GGML_OP_MUL_MAT && mt->src[1] == rs &&
                ggml_get_op_params_i32(mt, 1) == GGML_HINT_SRC0_IS_HADAMARD && mt->src[0]->ne[0] == 1024 &&
                g->type == GGML_TYPE_F32 && u->type == GGML_TYPE_F32 && node->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(g) && ggml_is_contiguous(u) && ggml_are_same_shape(g, u) &&
                ggml_are_same_shape(g, node) && g->ne[3] == 1 && sg && sg->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(sg) && sg->ne[0] == g->ne[0] && ggml_nrows(sg) == 1 && ggml_is_contiguous(mt) &&
                ggml_can_fuse_subgraph_ext(cgraph, gidx, 4, gops, gout, 1) &&
                ggml_sycl_same_or_disjoint(mt, g) && ggml_sycl_same_or_disjoint(mt, u) &&
                ggml_sycl_t2_fwht_quant(*sycl_ctx->stream(), nullptr, (const float *) sg->data, (float *) mt->data,
                                        ggml_nrows(g), g->ne[0], mt, (const float *) g->data, (const float *) u->data)) {
                static bool glu_logged = false;
                if (!glu_logged) {
                    glu_logged = true;
                    fprintf(stderr, "ggml_sycl: SWIGLU folded into the rotation (%s)\n", node->name);
                }
                i += 3;
                continue;
            }
        }
        // Hadamard sign flip + reshape + FWHT-hint matmul: apply the signs in the transform's load
        if (node->op == GGML_OP_MUL &&
            ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_MUL_MAT }, { i + 2 })) {
            const ggml_tensor * reshape = cgraph->nodes[i + 1];
            ggml_tensor *       mm      = cgraph->nodes[i + 2];
            const ggml_tensor * x       = node->src[0];
            const ggml_tensor * signs   = node->src[1];

            const bool pattern_ok = ggml_get_op_params_i32(mm, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
                mm->src[1] == reshape && reshape->src[0] == node && signs->ne[1] == 1 && signs->ne[2] == 1 &&
                signs->ne[3] == 1 && signs->type == GGML_TYPE_F32 &&
                (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16) && node->type == x->type &&
                ggml_is_contiguous(x) && ggml_is_contiguous(signs) && signs->ne[0] == x->ne[0] &&
                signs->ne[0] % mm->src[0]->ne[0] == 0;
            // ARC-LAB: the rotation that feeds T2 mat-vecs also quantizes for them (they read mm through a reshape)
            if (pattern_ok && x->type == GGML_TYPE_F32 && mm->src[0]->ne[0] == 1024 && x->ne[3] == 1 &&
                ggml_is_contiguous(mm) &&
                ggml_sycl_t2_fwht_quant(*sycl_ctx->stream(), (const float *) x->data, (const float *) signs->data,
                                        (float *) mm->data, ggml_nrows(x), x->ne[0], mm)) {
                i += 2;
                continue;
            }
            if (pattern_ok && ggml_sycl_op_fwht_signed(*sycl_ctx, x, signs, mm)) {
                i += 2;
                continue;
            }
        }
        if (node->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
            ggml_sycl_op_rms_norm_fused(*sycl_ctx, node, cgraph->nodes[i + 1]);
            i++;
            continue;
        }
        if (node->op == GGML_OP_UNARY &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { ggml_get_unary_op(node) })) {
            ggml_sycl_op_unary_mul_fused(*sycl_ctx, node, cgraph->nodes[i + 1]);
            i++;
            continue;
        }

        if (node->op == GGML_OP_MUL_MAT && ggml_sycl_mul_mat_glu_mmvq_fused(*sycl_ctx, cgraph, i)) {
            i += 2;
            continue;
        }
        if (node->op == GGML_OP_MUL_MAT && ggml_sycl_try_t2_multi(*sycl_ctx, cgraph, i, done, pending)) {
            continue;
        }
        if (node->op == GGML_OP_MUL_MAT && ggml_sycl_try_ptq1_multi(*sycl_ctx, cgraph, i, done, pending)) {
            continue;
        }

        bool ok = ggml_sycl_compute_forward(*sycl_ctx, node);
        if (!ok) {
            GGML_LOG_ERROR("%s: error: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
        }
        GGML_ASSERT(ok);
    }
    if (ot_on) {
        ot_mark(std::string());
        if (++g_ot.graphs == 64) {
            g_ot.report();
        }
    }
}

#ifdef GGML_SYCL_GRAPH
static bool check_graph_compatibility(ggml_cgraph * cgraph) {
    if (ggml_sycl_info().device_count > 1) {
        // A sycl_ex::command_graph object can only be created for a single device
        GGML_LOG_INFO("%s: disabling SYCL graphs due to multiple devices\n", __func__);
        return false;
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_op node_op = cgraph->nodes[i]->op;
        switch (node_op) {
            default:
                break;
            case GGML_OP_MUL_MAT_ID:
                // ggml_sycl_mul_mat_id() does a blocking host wait on the sycl queue after
                // submitting a memcpy operation, but wait() can't be called on a queue that
                // is recording to a graph.
                GGML_LOG_INFO("%s: disabling SYCL graphs due to unsupported node type %s\n", __func__,
                              ggml_op_name(node_op));
                return false;
            case GGML_OP_MUL_MAT:
                // oneMKL gemm (multi-token batches) cannot be recorded to a graph.
                if (cgraph->nodes[i]->src[1]->ne[1] > 1) {
                    return false;
                }
                // We cannot use graphs with ggml_sycl_mul_mat() when SYCL async memory allocation extensions are not available,
                // as SYCL malloc / free and host wait calls are not supported when recording to a graph which are all present
                // in reordering.
                if (!g_ggml_sycl_use_async_mem_op) {
                    GGML_LOG_INFO(
                        "%s: disabling SYCL graphs due to unsupported node type when using a compiler without the "
                        "oneAPI async memory allocation extension "
                        "%s\n",
                        __func__, ggml_op_name(node_op));
                    return false;
                }
        }
    }
    return true;
}
#endif

namespace {
std::mutex                           g_knob_mtx;
std::unordered_map<std::string, int> g_knob_file;   // from GGML_SYCL_KNOBS_FILE (wins)
std::unordered_map<std::string, int> g_knob_env;    // environment, read once per name
}  // namespace

int ggml_sycl_knob(const char * name, int def) {
    std::lock_guard<std::mutex> lock(g_knob_mtx);
    auto f = g_knob_file.find(name);
    if (f != g_knob_file.end()) {
        return f->second;
    }
    auto e = g_knob_env.find(name);
    if (e != g_knob_env.end()) {
        return e->second;
    }
    const char * v = getenv(name);
    const int    r = v ? atoi(v) : def;
    g_knob_env[name] = r;
    return r;
}

void ggml_sycl_knobs_refresh() {
    static const char * path = getenv("GGML_SYCL_KNOBS_FILE");
    if (!path) {
        return;
    }
    static std::filesystem::file_time_type last_time{};
    static std::uintmax_t                  last_size = 0;
    std::error_code                        ec;
    const auto                             t  = std::filesystem::last_write_time(path, ec);
    if (ec) {
        return;
    }
    const auto sz = std::filesystem::file_size(path, ec);
    if (ec || (t == last_time && sz == last_size)) {
        return;
    }
    last_time = t;
    last_size = sz;
    std::unordered_map<std::string, int> m;
    if (FILE * fp = fopen(path, "r")) {
        char line[256];
        while (fgets(line, sizeof(line), fp)) {
            char * eq = strchr(line, '=');
            if (!eq || line[0] == '#') {
                continue;
            }
            *eq = 0;
            std::string k(line);
            k.erase(k.find_last_not_of(" \t") + 1);
            m[k] = atoi(eq + 1);
        }
        fclose(fp);
    }
    std::string desc;
    for (auto & kv : m) {
        desc += " " + kv.first + "=" + std::to_string(kv.second);
    }
    GGML_LOG_INFO("%s: knobs reloaded from %s:%s\n", __func__, path, desc.c_str());
    std::lock_guard<std::mutex> lock(g_knob_mtx);
    g_knob_file = std::move(m);
}

static ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * sycl_ctx = static_cast<ggml_backend_sycl_context *>(backend->context);
    g_ht.in_big = g_ht.on && cgraph->n_nodes > 500;
    if (g_ht.in_big) {
        g_ht.t_in = sycl_host_timing::clk::now();
        if (g_ht.last_sync_end.time_since_epoch().count()) {
            g_ht.between += std::chrono::duration<double, std::milli>(g_ht.t_in - g_ht.last_sync_end).count();
        }
        g_ht.nodes += cgraph->n_nodes;
    }
    struct ht_exit {
        ~ht_exit() {
            if (g_ht.in_big) {
                g_ht.t_enq = sycl_host_timing::clk::now();
                g_ht.enqueue += std::chrono::duration<double, std::milli>(g_ht.t_enq - g_ht.t_in).count();
            }
        }
    } ht_exit_guard;
    ggml_sycl_knobs_refresh();
    ggml_sycl_t2_new_graph();

#ifdef GGML_SYCL_GRAPH
    bool use_sycl_graph = false;
    if (g_ggml_sycl_enable_graph) {
        use_sycl_graph = check_graph_compatibility(cgraph);
    }
    if (use_sycl_graph) {
        const bool graph_support = dpct::get_device(sycl_ctx->device).has(sycl::aspect::ext_oneapi_limited_graph);
        if (!graph_support) {
            GGML_SYCL_DEBUG("[SYCL-GRAPH] can not use graphs on device:%d\n", sycl_ctx->device);
            ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
            return GGML_STATUS_SUCCESS;
        }

        sycl_ex::command_graph model_sycl_graph(*(sycl_ctx->stream()), {sycl_ex::property::graph::assume_buffer_outlives_graph{}});

        model_sycl_graph.begin_recording(*(sycl_ctx->stream()));
        ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
        model_sycl_graph.end_recording();

        const bool graph_update_support = dpct::get_device(sycl_ctx->device).has(sycl::aspect::ext_oneapi_graph);
        if (!sycl_ctx->exec_graph || !graph_update_support) {
            auto exec_graph = graph_update_support ? model_sycl_graph.finalize(sycl_ex::property::graph::updatable{}) :
                                                     model_sycl_graph.finalize();
            sycl_ctx->exec_graph = std::make_unique<
                sycl_ex::command_graph<sycl_ex::graph_state::executable>>(exec_graph);
        } else {
            try {
                sycl_ctx->exec_graph->update(model_sycl_graph);
                GGML_SYCL_DEBUG("[SYCL-GRAPH] update success\n");
            } catch (sycl::exception const & e) {
                GGML_SYCL_DEBUG("[SYCL-GRAPH] Exception when updating graph, %s\n", e.what());
                auto exec_graph = model_sycl_graph.finalize({sycl_ex::property::graph::updatable{}});
                sycl_ctx->exec_graph = std::make_unique<
                    sycl_ex::command_graph<sycl_ex::graph_state::executable>>(exec_graph);
            }
        }

        sycl_ctx->stream()->ext_oneapi_graph(*(sycl_ctx->exec_graph));
    } else
#endif
    {
        ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_sycl_event_record(ggml_backend_t backend, ggml_backend_event_t event)
try
{
    ggml_backend_sycl_context *sycl_ctx =
        (ggml_backend_sycl_context *)backend->context;

    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);

    const queue_ptr &stream = sycl_ctx->stream(sycl_ctx->device, 0);
    // Record the current state of the queue
    SYCL_CHECK(CHECK_TRY_ERROR(*sycl_event = stream->ext_oneapi_submit_barrier()));
}
catch (sycl::exception const &exc)
{
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static void ggml_backend_sycl_event_wait(ggml_backend_t backend, ggml_backend_event_t event) try {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    sycl::event* sycl_event = static_cast<sycl::event*>(event->context);

    if (ggml_backend_is_sycl(backend)) {
        SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->wait()));
    } else
        GGML_ABORT("fatal error");
} catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static ggml_backend_i ggml_backend_sycl_interface = {
    /* .get_name                = */ ggml_backend_sycl_get_name,
    /* .free                    = */ ggml_backend_sycl_free,
    /* .set_tensor_async        = */ ggml_backend_sycl_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_sycl_get_tensor_async,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL, // ggml_backend_sycl_cpy_tensor_async,
                                           // // TODO: update for the new
                                           // interface
    /* .synchronize             = */ ggml_backend_sycl_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_sycl_graph_compute,
    /* .event_record            = */ ggml_backend_sycl_event_record,
    /* .event_wait              = */ ggml_backend_sycl_event_wait,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_sycl_guid() {
    static ggml_guid guid = { 0x58, 0x05, 0x13, 0x8f, 0xcd, 0x3a, 0x61, 0x9d, 0xe7, 0xcd, 0x98, 0xa9, 0x03, 0xfd, 0x7c, 0x53 };
    return &guid;
}

bool ggml_backend_is_sycl(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_sycl_guid());
}

int ggml_backend_sycl_get_device_count() {
    return ggml_sycl_info().device_count;
}


// backend device

static const char * ggml_backend_sycl_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_sycl_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_sycl_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    ggml_sycl_set_device(ctx->device);
    SYCL_CHECK(CHECK_TRY_ERROR(
    dpct::dev_mgr::instance().get_device(ctx->device).get_memory_info(*free, *total)));
}

static enum ggml_backend_dev_type ggml_backend_sycl_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    const sycl_device_info & info = ggml_sycl_info().devices[ctx->device];
    if (info.l0_device_type_valid && !info.l0_discrete_gpu) {
        return GGML_BACKEND_DEVICE_TYPE_IGPU;
    }
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_sycl_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_sycl_device_get_name(dev);
    props->description = ggml_backend_sycl_device_get_description(dev);
    props->type        = ggml_backend_sycl_device_get_type(dev);
    ggml_backend_sycl_device_get_memory(dev, &props->memory_free, &props->memory_total);

    bool host_buffer = getenv("GGML_SYCL_NO_PINNED") == nullptr;
#ifdef GGML_SYCL_NO_PEER_COPY
    bool events = false;
#else
    bool events = true;
#endif

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ events,
        /* .mmap_support          = */ true,
    };
}

static ggml_backend_t ggml_backend_sycl_device_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    return ggml_backend_sycl_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_sycl_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    return ggml_backend_sycl_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_sycl_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_sycl_host_buffer_type();
}

static ggml_backend_buffer_t ggml_backend_sycl_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(ptr);
    GGML_UNUSED(size);
    GGML_UNUSED(max_tensor_size);
    return nullptr;
}

static bool do_ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_sycl_device_context *sycl_ctx =
        (ggml_backend_sycl_device_context *)dev->context;
    int device = sycl_ctx->device;
    switch (op->op) {
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            }
        case GGML_OP_CONV_2D:
        case GGML_OP_CONV_2D_DW:
        case GGML_OP_CONV_TRANSPOSE_2D:
            return true;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_XIELU:
                case GGML_UNARY_OP_CEIL:
                    return true;
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    return true;
                default:
                    return false;
            }
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                    return ggml_is_contiguous_1(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                struct ggml_tensor * a = op->src[0];
                struct ggml_tensor * b = op->src[1];

                if (a->ne[3] != b->ne[3]) {
                    return false;
                }

                ggml_type src0_type = op->src[0]->type;

                // TODO: The configuration below needs more work to be supported with oneDNN
                if (ggml_is_permuted(a) && !ggml_is_contiguous(a) &&
                    a->ne[2] > 1 && a->ne[3] > 1 && src0_type == GGML_TYPE_F16) {
                  return false;
                }

                // TODO: This specific configuration can fail with oneDNN and needs more debugging
                if (!ggml_is_permuted(a) && ggml_is_permuted(b) && b->ne[2] > 1 && b->ne[3] > 1 &&
                    a->ne[0] > 128 && a->ne[2] == 1 && src0_type == GGML_TYPE_F16) {
                    return false;
                }

                if (ggml_is_quantized(src0_type) &&
                    !ggml_sycl_supports_mmvq(src0_type) &&
                    !ggml_sycl_supports_dmmv(src0_type) &&
                    !ggml_sycl_supports_mmq(src0_type)) {
                    return false;
                }

                return true;
            }
        case GGML_OP_OUT_PROD:
            return op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F32 ||
                    (op->src[0]->type == GGML_TYPE_Q1_0 && op->src[0]->ne[2] == op->src[1]->ne[2] &&
                     op->src[0]->ne[3] == op->src[1]->ne[3])) &&
                   op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_I32:
                    case GGML_TYPE_PTQ1_0:  // ARC-LAB: dequantize_ptq1_0_f32, bit-exact with the CPU
                    case GGML_TYPE_F16:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_PQ2_0:
                    case GGML_TYPE_MXFP4:
                    case GGML_TYPE_NVFP4:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_0:
                        return true;
                    default:
                        return false;
                }
            }
         case GGML_OP_SET:
               return (op->type == GGML_TYPE_F32) &&
                      (op->src[0] && op->src[1]) &&
                      (op->src[0]->type == GGML_TYPE_F32) &&
                      (op->src[1]->type == GGML_TYPE_F32);

        case GGML_OP_SET_ROWS:
            {
                auto res = (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 ||
                            op->src[0]->type == GGML_TYPE_BF16) &&
                           (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
                return res;
            }
            break;
        case GGML_OP_DSV4_HC_PRE:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_COMB:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_POST:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->src[3]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_LIGHTNING_INDEXER:
            return op->src[0]->type == GGML_TYPE_F32 &&
                (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32 ||
                 op->src[1]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_Q8_0 ||
                 op->src[1]->type == GGML_TYPE_Q5_1 || op->src[1]->type == GGML_TYPE_Q5_0 ||
                 op->src[1]->type == GGML_TYPE_Q4_1 || op->src[1]->type == GGML_TYPE_Q4_0 ||
                 op->src[1]->type == GGML_TYPE_IQ4_NL) &&
                op->src[2]->type == GGML_TYPE_F32 &&
                op->src[3]->type == GGML_TYPE_F16 &&
                op->type == GGML_TYPE_F32 &&
                op->src[0]->ne[0] == WARP_SIZE * 8;
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;

                // Quantizing a float row into PTQ1_0 or PQ2_0 has no kernel: both are
                // produced offline by the converter, which also applies the Hadamard
                // rotation the packing assumes. ggml_sycl_cpy() would take the
                // float -> quantized branch and assert, so decline the pair here and let
                // the scheduler fall back. The quant -> same-quant copies are handled.
                if ((src1_type == GGML_TYPE_PTQ1_0 || src1_type == GGML_TYPE_PQ2_0) &&
                    src0_type != src1_type) {
                    return false;
                }

                if (src0_type == GGML_TYPE_F16) {
                    if (src1_type == GGML_TYPE_Q2_K ||
                        src1_type == GGML_TYPE_Q3_K ||
                        src1_type == GGML_TYPE_Q4_K ||
                        src1_type == GGML_TYPE_Q5_K ||
                        src1_type == GGML_TYPE_Q6_K ||
                        src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ4_XS) {
                        return false;
                    }
                }

                if (src0_type == GGML_TYPE_BF16) {
                    if (src1_type == GGML_TYPE_Q4_0 || //big error in ut
                        src1_type == GGML_TYPE_Q4_1 || //big error in ut
                        src1_type == GGML_TYPE_Q8_0 || //big error in ut
                        src1_type == GGML_TYPE_Q2_K ||
                        src1_type == GGML_TYPE_Q3_K ||
                        src1_type == GGML_TYPE_Q4_K ||
                        src1_type == GGML_TYPE_Q5_K ||
                        src1_type == GGML_TYPE_Q6_K ||
                        src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ4_XS) {
                        return false;
                    }
                }

                if (src0_type == GGML_TYPE_F32) {
                    if (src1_type == GGML_TYPE_Q2_K ||
                        src1_type == GGML_TYPE_Q3_K ||
                        src1_type == GGML_TYPE_Q4_K ||
                        src1_type == GGML_TYPE_Q5_K ||
                        src1_type == GGML_TYPE_Q6_K ||
                        src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ4_XS) {
                        return false;
                    }
                }

                if (src1_type == GGML_TYPE_F32) {
                    if (src0_type == GGML_TYPE_Q1_0 ||
                        src0_type == GGML_TYPE_PTQ1_0 ||
                        src0_type == GGML_TYPE_PQ2_0 ||
                        src0_type == GGML_TYPE_NVFP4 ||
                        src0_type == GGML_TYPE_Q2_K ||
                        src0_type == GGML_TYPE_Q3_K ||
                        src0_type == GGML_TYPE_Q4_K ||
                        src0_type == GGML_TYPE_Q5_K ||
                        src0_type == GGML_TYPE_Q6_K ||
                        src0_type == GGML_TYPE_IQ2_XXS ||
                        src0_type == GGML_TYPE_IQ2_XS ||
                        src0_type == GGML_TYPE_IQ2_S ||
                        src0_type == GGML_TYPE_IQ3_XXS ||
                        src0_type == GGML_TYPE_IQ1_S ||
                        src0_type == GGML_TYPE_IQ1_M ||
                        src0_type == GGML_TYPE_IQ3_S ||
                        src0_type == GGML_TYPE_IQ4_NL ||
                        src0_type == GGML_TYPE_IQ4_XS
                    ) {
                        return false;
                    }
                }

                if (src0_type == src1_type) {
                    if (src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M) {
                        return false;
                    }
                }

                return true;
            }
        case GGML_OP_REPEAT_BACK:
            {
                ggml_type src0_type = op->src[0]->type;
                return src0_type == GGML_TYPE_F32;
            }
        case GGML_OP_CONCAT:
        case GGML_OP_DUP:
        case GGML_OP_ARGMAX:
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD:
        case GGML_OP_ADD1:
        case GGML_OP_ADD_ID:
        case GGML_OP_SUB:
        case GGML_OP_COUNT_EQUAL:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_REPEAT:
            return true;
        case GGML_OP_PAD_REFLECT_1D:
            return ggml_is_contiguous(op->src[0]) && op-> type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_CLAMP:
        case GGML_OP_LOG:
        case GGML_OP_NORM:
        case GGML_OP_L2_NORM:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_RMS_NORM:
            return true;
        case GGML_OP_RMS_NORM_BACK:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_SCALE:
            return true;
        case GGML_OP_CONT:
            return true;
        case GGML_OP_TRI:
            {
                const ggml_tensor * src0 = op->src[0];
                return src0 &&
                       op->type == GGML_TYPE_F32 &&
                       ggml_is_contiguous(src0);
            }
        case GGML_OP_DIAG_MASK_INF:
            return true;
        case GGML_OP_SOFT_MAX:
            return true;
        case GGML_OP_SOFT_MAX_BACK: {
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
            return max_bias == 0.0f;
        }
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_UPSCALE:
            return true;
        case GGML_OP_COL2IM_1D:
            return ggml_is_contiguous(op->src[0]) &&
                   (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16
#ifdef GGML_SYCL_HAS_BF16
                    || op->type == GGML_TYPE_BF16
#endif
                   ) &&
                   op->src[0]->type == op->type;
        case GGML_OP_CONV_3D:
            return op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op->src[1]);
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_ARGSORT:
            return true;
        case GGML_OP_TOP_K: {
            const ggml_tensor * src0 = op->src[0];
            const int k = op->ne[0];
            return src0 &&
                op->type == GGML_TYPE_I32 &&
                src0->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(src0) &&
                k > 0 && k <= 32;
        }
        case GGML_OP_POOL_2D:
        case GGML_OP_POOL_1D:
        case GGML_OP_ACC:
            return true;
        case GGML_OP_PAD:
            if (ggml_get_op_params_i32(op, 8) != 0) {
                return false;
            }
            return true;
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_RWKV_WKV7:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
            return true;
        case GGML_OP_GATED_DELTA_NET:
            // rows-indexed state read (src[6]) and raw gates (op_params[1]) not implemented here
            return op->src[6] == NULL && ggml_get_op_params_i32(op, 1) == 0;
        case GGML_OP_GATED_DELTA_NET_REPLAY:
            // ARC-LAB: kernels exist for the scalar gate at head size 128 (Bonsai / Qwen3.5) only
            return op->src[2]->ne[0] == 128 && op->src[3]->ne[0] == 1 && ggml_are_same_stride(op->src[0], op->src[1]);
        case GGML_OP_SSM_CONV:
            return op->type == GGML_TYPE_F32 &&
                   op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_ROLL:
            return op->type == GGML_TYPE_F32;
        case GGML_OP_ARANGE:
            return op->type == GGML_TYPE_F32;
        case GGML_OP_SSM_SCAN:
            if (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 128 || d_state == 256) && d_head % WARP_SIZE == 0)
                return (op->src[0]->ne[0] == 128 || op->src[0]->ne[0] == 256) && op->src[0]->ne[1] % WARP_SIZE == 0;
            } else {
                // TODO Mamba-1 not yet ported to SYCL
                return false;
            }
        case GGML_OP_FILL:
        case GGML_OP_CUMSUM:
        case GGML_OP_DIAG:
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            return true;
        case GGML_OP_SOLVE_TRI:
            return op->src[0]->ne[0] <= SYCL_SOLVE_TRI_MAX_N && op->src[1]->ne[0] <= SYCL_SOLVE_TRI_MAX_K;
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_sycl_flash_attn_ext_supported(device, op);
        default:
            return false;
    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    bool res = do_ggml_backend_sycl_device_supports_op(dev, op);
    GGML_SYCL_DEBUG("[SYCL] call %s op->op=%s op->type=%s -> %s\n", __func__, ggml_op_name(op->op),
                    ggml_type_name(op->type), res ? "true" : "false");
    return res;
}

static bool ggml_backend_sycl_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (buft->iface.get_name != ggml_backend_sycl_buffer_type_get_name) {
        return false;
    }
    ggml_backend_sycl_buffer_type_context * buft_ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;
    ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
    return buft_ctx->device == sycl_ctx->device;
}

static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_GET_ROWS:
            return 0;
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_sycl_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
    return get_op_batch_size(op) >= sycl_ctx->op_offload_min_batch_size;
}

static ggml_backend_event_t
ggml_backend_sycl_device_event_new(ggml_backend_dev_t dev) {

#ifdef GGML_SYCL_NO_PEER_COPY
    return nullptr;
#else
  sycl::event *event_ptr = new sycl::event();

  return new ggml_backend_event{
      /* .device = */ dev,
      /* .context = */ event_ptr,
  };
#endif
}

static void ggml_backend_sycl_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) try {
  GGML_UNUSED(dev);
  if (event == nullptr) {
    return;
  }

  if (event->context != nullptr) {
    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
    delete sycl_event;
    event->context = nullptr;
  }

  delete event;
} catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}


static void ggml_backend_sycl_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) try {
  GGML_UNUSED(dev);
  GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);

  sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
  SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->wait()));
} catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static const ggml_backend_device_i ggml_backend_sycl_device_interface = {
    /* .get_name                = */ ggml_backend_sycl_device_get_name,
    /* .get_description         = */ ggml_backend_sycl_device_get_description,
    /* .get_memory              = */ ggml_backend_sycl_device_get_memory,
    /* .get_type                = */ ggml_backend_sycl_device_get_type,
    /* .get_props               = */ ggml_backend_sycl_device_get_props,
    /* .init_backend            = */ ggml_backend_sycl_device_init,
    /* .get_buffer_type         = */ ggml_backend_sycl_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_sycl_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ ggml_backend_sycl_device_buffer_from_host_ptr,
    /* .supports_op             = */ ggml_backend_sycl_device_supports_op,
    /* .supports_buft           = */ ggml_backend_sycl_device_supports_buft,
    /* .offload_op              = */ ggml_backend_sycl_device_offload_op,
    /* .event_new               = */ ggml_backend_sycl_device_event_new,
    /* .event_free              = */ ggml_backend_sycl_device_event_free,
    /* .event_synchronize       = */ ggml_backend_sycl_device_event_synchronize,
};

// backend reg

struct ggml_backend_sycl_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_sycl_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_SYCL_NAME;
}

static size_t ggml_backend_sycl_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_sycl_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

// ==========================================================================
// Tensor parallelism (--split-mode tensor) for the SYCL backend.
//
// The meta-backend invokes these three entry points via get_proc_address:
//   * ggml_backend_sycl_comm_init             - one-time per-graph setup
//   * ggml_backend_sycl_comm_allreduce_tensor - per-allreduce step
//   * ggml_backend_sycl_comm_free             - tear-down
//
// For N=2 (dual-GPU), this is a degenerate ring allreduce with dual paths
// chosen by tensor size:
//
//   * Small (nelem < 32K): FP32 direct memcpy + per-device ADD
//     kernel. The kernel depends_on() its corresponding memcpy event
//     so it doesn't read partial data. Both devices run in parallel.
//
//   * Large (nelem >= 32K): BF16-compressed. Each device compresses
//     its FP32 partial to BF16 locally, cross-device memcpys
//     to the peer (half the PCI bandwidth), where it is decompressed
//     and added into the local FP32 partial. 6 SYCL submissions per
//     allreduce (2 compress + 2 memcpy + 2 decompress-add) vs the
//     4 for the small path, but the bandwidth saving > 6 GB/s PCIe x 2
//     dominates for larger tensors.
//
// Storage: A persistent uint8_t buffer per device, sized to
// 4 * nelem bytes. Both paths reinterpret the same bytes (small path
// as nelem floats; large path as outbox + inbox = 2*nelem uint16_t
// each, using the full 4*nelem byte budget either way). Single
// alloc+free per device keeps the SYCL pool's strict-LIFO invariant
// trivial.
//
// For non-(N=2 FP32 contiguous) cases, comm_init or comm_allreduce_tensor
// returns null/false, causing the meta-backend to use its generic
// butterfly all-reduce fallback.
// ==========================================================================

struct ggml_backend_sycl_comm_context {
    std::vector<ggml_backend_t> backends;
    // ONE persistent per-device byte buffer, 4*nelem bytes.  Both the
    // FP32 small-tensor path and the BF16 large-tensor path share it
    // by reinterpreting.
    std::unique_ptr<ggml_sycl_pool_alloc<uint8_t>> buf0;
    std::unique_ptr<ggml_sycl_pool_alloc<uint8_t>> buf1;
    int64_t buf_nelem = 0;
};

void * ggml_backend_sycl_comm_init(ggml_backend_t * backends, size_t n_backends) try {
    for (size_t i = 0; i < n_backends; ++i) {
        if (!ggml_backend_is_sycl(backends[i])) {
            return nullptr;
        }
    }

    // Initial version: N=2 only. For N!=2, returning null makes the
    // meta-backend skip this backend-specific allreduce entirely.
    if (n_backends != 2) {
        return nullptr;
    }

    auto * ctx = new ggml_backend_sycl_comm_context;
    ctx->backends.assign(backends, backends + n_backends);
    auto * sctx0 = (ggml_backend_sycl_context *) backends[0]->context;
    auto * sctx1 = (ggml_backend_sycl_context *) backends[1]->context;
    ctx->buf0 = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(sctx0->pool());
    ctx->buf1 = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(sctx1->pool());
    return ctx;
}
catch (const sycl::exception &) { return nullptr; }
catch (...)                     { return nullptr; }

void ggml_backend_sycl_comm_free(void * comm_ctx_v) {
    auto * comm_ctx = static_cast<ggml_backend_sycl_comm_context *>(comm_ctx_v);
    if (comm_ctx == nullptr) {
        return;
    }

    // Sync both per-device queues so the pool_alloc destructors don't
    // return memory still in use by the last kernel.
    if (comm_ctx->backends.size() == 2) {
        auto * sctx0 = (ggml_backend_sycl_context *) comm_ctx->backends[0]->context;
        auto * sctx1 = (ggml_backend_sycl_context *) comm_ctx->backends[1]->context;
        try {
            sctx0->stream()->wait();
            sctx1->stream()->wait();
        } catch (...) { /* best effort during shutdown */ }
    }

    delete comm_ctx;
}

bool ggml_backend_sycl_comm_allreduce_tensor(void * comm_ctx_v, struct ggml_tensor ** tensors) try {
    if (comm_ctx_v == nullptr) {
        return false;
    }

    auto * comm_ctx = static_cast<ggml_backend_sycl_comm_context *>(comm_ctx_v);
    const size_t n_backends = comm_ctx->backends.size();

    // Fast path: N=2, F32/F16, contiguous, matching shapes.
    if (n_backends != 2) {
        return false;
    }
    // Accept F32 or F16 inputs natively (types must match). F16 takes the
    // direct 2-byte memcpy + add path below; other types return false so the
    // meta-backend uses its generic all-reduce.
    if (tensors[0]->type != tensors[1]->type) {
        return false;
    }
    if (tensors[0]->type != GGML_TYPE_F32 && tensors[0]->type != GGML_TYPE_F16) {
        return false;
    }
    if (!ggml_is_contiguous(tensors[0]) || !ggml_is_contiguous(tensors[1])) {
        return false;
    }
    if (ggml_nelements(tensors[0]) != ggml_nelements(tensors[1])) {
        return false;
    }

    const int64_t nelem  = ggml_nelements(tensors[0]);
    const size_t  nbytes = ggml_nbytes(tensors[0]);
    if (nelem == 0) {
        return true;
    }

    auto * ctx0 = (ggml_backend_sycl_context *) comm_ctx->backends[0]->context;
    auto * ctx1 = (ggml_backend_sycl_context *) comm_ctx->backends[1]->context;
    queue_ptr q0 = ctx0->stream();
    queue_ptr q1 = ctx1->stream();

    // Grow per-device byte buffers if needed (4 * nelem bytes each).
    if (comm_ctx->buf_nelem < nelem) {
        comm_ctx->buf0->realloc(nelem * 4);
        comm_ctx->buf1->realloc(nelem * 4);
        comm_ctx->buf_nelem = nelem;
    }
    uint8_t * buf0 = comm_ctx->buf0->get();
    uint8_t * buf1 = comm_ctx->buf1->get();

    // F16 native path: direct 2-byte cross-device copy + add, skipping the
    // F32 round-trip the meta-backend fallback would force. Cross-device copies
    // go through dev2dev_memcpy because the two devices are in separate SYCL
    // contexts (a raw peer-USM q->memcpy would be a silent no-op).
    if (tensors[0]->type == GGML_TYPE_F16) {
        sycl::half * f16_out0 = (sycl::half *) tensors[0]->data;
        sycl::half * f16_out1 = (sycl::half *) tensors[1]->data;
        sycl::half * f16_tmp0 = (sycl::half *) buf0;
        sycl::half * f16_tmp1 = (sycl::half *) buf1;

        q0->wait();
        q1->wait();
        dev2dev_memcpy(ctx0->device, *q0, ctx1->device, *q1, f16_tmp0, tensors[1]->data, nbytes);
        dev2dev_memcpy(ctx1->device, *q1, ctx0->device, *q0, f16_tmp1, tensors[0]->data, nbytes);

        q0->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                f16_out0[i] = (sycl::half) ((float) f16_out0[i] + (float) f16_tmp0[i]);
            });
        });
        q1->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                f16_out1[i] = (sycl::half) ((float) f16_out1[i] + (float) f16_tmp1[i]);
            });
        });
        return true;
    }

    float * out0 = (float *) tensors[0]->data;
    float * out1 = (float *) tensors[1]->data;

    // BF16 threshold: above this, the PCIe savings from halving the
    // cross-device bytes outweigh the 2 extra compress kernels.
    // Below: stay on the FP32 fast path.  Threshold mirrors the CUDA
    // NCCL allreduce pattern for n_backends=2.
    static constexpr int64_t BF16_THRESHOLD = 32768;

    if (nelem < BF16_THRESHOLD) {
        // FP32 small path: 4 SYCL submissions per allreduce.
        float * tmp0 = (float *) buf0;
        float * tmp1 = (float *) buf1;

        // COMM-D2D-FIX: the two devices are in SEPARATE SYCL contexts, so a raw
        // q->memcpy of a peer USM pointer is a silent no-op. Route cross-device
        // copies through dev2dev_memcpy (L0 direct copy / host staging). It is
        // synchronous, so wait for the local partials to be produced first.
        q0->wait();
        q1->wait();
        dev2dev_memcpy(ctx0->device, *q0, ctx1->device, *q1, tmp0, tensors[1]->data, nbytes);
        dev2dev_memcpy(ctx1->device, *q1, ctx0->device, *q0, tmp1, tensors[0]->data, nbytes);

        q0->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                out0[i] += tmp0[i];
            });
        });
        q1->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                out1[i] += tmp1[i];
            });
        });
        return true;
    }

    // BF16 large path: 6 SYCL submissions per allreduce, but the
    // cross-device memcpy is HALF the bytes. Pure bit-shift
    // conversion (no rounding) — matches ggml's truncating fp32->bf16.
    uint16_t * outbox0 = (uint16_t *) buf0;
    uint16_t * inbox0  = outbox0 + nelem;
    uint16_t * outbox1 = (uint16_t *) buf1;
    uint16_t * inbox1  = outbox1 + nelem;

    // Phase A: compress each device's local partial in parallel.
    sycl::event c0 = q0->parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
        outbox0[i] = (uint16_t) (sycl::bit_cast<uint32_t>(out0[i]) >> 16);
    });

    sycl::event c1 = q1->parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
        outbox1[i] = (uint16_t) (sycl::bit_cast<uint32_t>(out1[i]) >> 16);
    });

    // Phase B: COMM-D2D-FIX-BF16 cross-device copy of compressed bytes via
    // dev2dev_memcpy (separate SYCL contexts; sync copy after compress).
    const size_t bf16_bytes = nelem * sizeof(uint16_t);
    c0.wait();
    c1.wait();
    dev2dev_memcpy(ctx0->device, *q0, ctx1->device, *q1, inbox0, outbox1, bf16_bytes);
    dev2dev_memcpy(ctx1->device, *q1, ctx0->device, *q0, inbox1, outbox0, bf16_bytes);

    // Phase C: decompress + add into local FP32 partial.
    q0->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
            out0[i] += sycl::bit_cast<float>(((uint32_t) inbox0[i]) << 16);
        });
    });

    q1->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
            out1[i] += sycl::bit_cast<float>(((uint32_t) inbox1[i]) << 16);
        });
    });

    return true;
}
catch (const sycl::exception &) { return false; }
catch (...)                     { return false; }

static void *ggml_backend_sycl_reg_get_proc_address(ggml_backend_reg_t reg, const char *name) {
    GGML_UNUSED(reg);

    if (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        return (void *)ggml_backend_sycl_split_buffer_type;
    }

    // Tensor parallelism (--split-mode tensor) entry points.
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *)ggml_backend_sycl_comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *)ggml_backend_sycl_comm_free;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *)ggml_backend_sycl_comm_allreduce_tensor;
    }

    // SYCL doesn't support registering host memory, left here for reference
    // "ggml_backend_register_host_buffer"
    // "ggml_backend_unregister_host_buffer"
    GGML_UNUSED(name);
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_sycl_reg_interface = {
    /* .get_name          = */ ggml_backend_sycl_reg_get_name,
    /* .get_device_count  = */ ggml_backend_sycl_reg_get_device_count,
    /* .get_device        = */ ggml_backend_sycl_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_sycl_reg_get_proc_address,
};


// backend registry

ggml_backend_reg_t ggml_backend_sycl_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_sycl_reg_context * ctx = new ggml_backend_sycl_reg_context;
            const int min_batch_size = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            for (int i = 0; i < ggml_sycl_info().device_count; i++) {
                ggml_backend_sycl_device_context * dev_ctx = new ggml_backend_sycl_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_SYCL_NAME + std::to_string(i);

                ggml_sycl_set_device(i);

                dpct::device_info prop;
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
                    prop, dpct::dev_mgr::instance().get_device(i))));

                dev_ctx->description = prop.get_name();
                dev_ctx->op_offload_min_batch_size = min_batch_size;

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface       = */ ggml_backend_sycl_device_interface,
                    /* .reg         = */ &reg,
                    /* .context     = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_sycl_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_sycl_init(int device) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_init\n");
    ggml_check_sycl();

    check_allow_gpu_index(device);

    ggml_backend_sycl_context * ctx = new ggml_backend_sycl_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return nullptr;
    };

    ggml_backend_t sycl_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_sycl_guid(),
        /* .iface   = */ ggml_backend_sycl_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), device),
        /* .context = */ ctx
    };

    return sycl_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_sycl_reg)
