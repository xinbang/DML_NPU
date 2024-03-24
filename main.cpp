// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "pch.h"

#include <dxcore_interface.h>
#include <dxcore.h>

#include "onnxruntime_cxx_api.h"
#include "dml_provider_factory.h"

#include "TensorHelper.h"

using Microsoft::WRL::ComPtr;

void InitializeDirectML(ID3D12Device1** d3dDeviceOut, ID3D12CommandQueue** commandQueueOut, IDMLDevice** dmlDeviceOut) {
    // Whether to skip adapters which support Graphics in order to target NPU for testing
    bool forceComputeOnlyDevice = true;
    ComPtr<IDXCoreAdapterFactory> factory;
    HMODULE dxCoreModule = LoadLibraryW(L"DXCore.dll");
    if (dxCoreModule)
    {
        auto dxcoreCreateAdapterFactory = reinterpret_cast<HRESULT(WINAPI*)(REFIID, void**)>(
            GetProcAddress(dxCoreModule, "DXCoreCreateAdapterFactory")
            );
        if (dxcoreCreateAdapterFactory)
        {
            dxcoreCreateAdapterFactory(IID_PPV_ARGS(&factory));
        }
    }
    // Create the DXCore Adapter
    ComPtr<IDXCoreAdapter> adapter;
    if (factory)
    {
        const GUID dxGUIDs[] = { DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE };
        ComPtr<IDXCoreAdapterList> adapterList;
        THROW_IF_FAILED(factory->CreateAdapterList(ARRAYSIZE(dxGUIDs), dxGUIDs, IID_PPV_ARGS(&adapterList)));
        for (uint32_t i = 0, adapterCount = adapterList->GetAdapterCount(); i < adapterCount; i++)
        {
            ComPtr<IDXCoreAdapter> nextGpuAdapter;
            THROW_IF_FAILED(adapterList->GetAdapter(static_cast<uint32_t>(i), IID_PPV_ARGS(&nextGpuAdapter)));
            if (!forceComputeOnlyDevice || !nextGpuAdapter->IsAttributeSupported(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS))
            {
                adapter = std::move(nextGpuAdapter);
                break;
            }
        }
    }
    // Create the D3D12 Device
    ComPtr<ID3D12Device1> d3dDevice;
    if (adapter)
    {
        HMODULE d3d12Module = LoadLibraryW(L"d3d12.dll");
        if (d3d12Module)
        {
            auto d3d12CreateDevice = reinterpret_cast<HRESULT(WINAPI*)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void*)>(
                GetProcAddress(d3d12Module, "D3D12CreateDevice")
                );
            if (d3d12CreateDevice)
            {
                THROW_IF_FAILED(d3d12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_1_0_CORE, IID_PPV_ARGS(&d3dDevice)));
            }
        }
    }
    // Create the DML Device and D3D12 Command Queue
    ComPtr<IDMLDevice> dmlDevice;
    ComPtr<ID3D12CommandQueue> commandQueue;
    if (d3dDevice)
    {
        D3D12_COMMAND_QUEUE_DESC queueDesc = {};
        queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        THROW_IF_FAILED(d3dDevice->CreateCommandQueue(
            &queueDesc,
            IID_PPV_ARGS(commandQueue.ReleaseAndGetAddressOf())));
        HMODULE dmlModule = LoadLibraryW(L"DirectML.dll");
        if (dmlModule)
        {
            auto dmlCreateDevice = reinterpret_cast<HRESULT(WINAPI*)(ID3D12Device*, DML_CREATE_DEVICE_FLAGS, DML_FEATURE_LEVEL, REFIID, void*)>(
                GetProcAddress(dmlModule, "DMLCreateDevice1")
                );
            if (dmlCreateDevice)
            {
                THROW_IF_FAILED(dmlCreateDevice(d3dDevice.Get(), DML_CREATE_DEVICE_FLAG_NONE, DML_FEATURE_LEVEL_5_0, IID_PPV_ARGS(dmlDevice.ReleaseAndGetAddressOf())));
            }
        }
    }

    d3dDevice.CopyTo(d3dDeviceOut);
    commandQueue.CopyTo(commandQueueOut);
    dmlDevice.CopyTo(dmlDeviceOut);
}

void main(int argc, char* argv[])
{
    if (argc < 2)
    {
        printf("Please use the commandline:DMLBench.exe modelfile Iterations(100) device(NPU under developing)\n");
        auto all_providers = Ort::GetAvailableProviders();
        bool support_dml = false;
        std::string providers_msg = "Your device supports below EP:\n";
        for (size_t i = 0; i < all_providers.size(); ++i) {
            providers_msg = providers_msg + all_providers[i] + ", ";
            if (all_providers[i] == "DmlExecutionProvider") {
                support_dml = true;
            }
        }
        printf("%s\n", providers_msg.c_str());
       return;
    }

    const size_t cSize = strlen(argv[1]) + 1;
    size_t converted = 0;
    wchar_t* modelFile = new wchar_t[cSize];
    mbstowcs_s(&converted, modelFile,cSize, argv[1], _TRUNCATE);

    int fenceValueStart = 2;
    int numIterations = 100;
    if (argc > 2)
    {
        numIterations = atoi(argv[2]);
    }

    ComPtr<ID3D12Device1> d3dDevice;
    ComPtr<IDMLDevice> dmlDevice;
    ComPtr<ID3D12CommandQueue> commandQueue;
    InitializeDirectML(d3dDevice.GetAddressOf(), commandQueue.GetAddressOf(), dmlDevice.GetAddressOf());

    // Add the DML execution provider to ORT using the DML Device and D3D12 Command Queue created above.
    if (!dmlDevice)
    {
        printf("No NPU device found\n");
        return;
    }

    const OrtApi& ortApi = Ort::GetApi();
    static Ort::Env s_OrtEnv{ nullptr };
    s_OrtEnv = Ort::Env(Ort::ThreadingOptions{});
    s_OrtEnv.DisableTelemetryEvents();

    auto sessionOptions = Ort::SessionOptions{};
    sessionOptions.DisableMemPattern();
    sessionOptions.DisablePerSessionThreads();
    sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    const OrtDmlApi* ortDmlApi = nullptr;
    Ort::ThrowOnError(ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ortDmlApi)));
    Ort::ThrowOnError(ortDmlApi->SessionOptionsAppendExecutionProvider_DML1(sessionOptions, dmlDevice.Get(), commandQueue.Get()));

    // Create the session
    auto startModelLoading = std::chrono::system_clock::now();
    auto session = Ort::Session(s_OrtEnv, modelFile, sessionOptions);
    auto endModelLoading = std::chrono::system_clock::now();
    printf("load model Took: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(endModelLoading - startModelLoading).count() / 1000.f);

    const char* inputName = "input";
    const char* outputName = "output";
    {
        size_t num_input_nodes = session.GetInputCount();
        size_t num_output_nodes = session.GetOutputCount();
        printf("number of input node is %zd\n", num_input_nodes);
        printf("number of output node is %zd\n", num_output_nodes);
        for (auto i = 0; i < num_input_nodes; i++)
        {
            std::vector<int64_t> input_dims = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            printf("input %d dim is: ", i);
            for (auto j = 0; j < input_dims.size(); j++)
                printf("%zd ", input_dims[j]);
        }
        for (auto i = 0; i < num_output_nodes; i++)
        {
            std::vector<int64_t> output_dims = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            printf("output %d dim is: ", i);
            for (auto j = 0; j < output_dims.size(); j++)
                printf("%zd ", output_dims[j]);
        }
        printf("\n");
    }
    // Create input tensor
    Ort::TypeInfo type_info = session.GetInputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    auto input = CreateDmlValue(tensor_info, commandQueue.Get());
    auto inputTensor = std::move(input.first);
    
    const auto memoryInfo = inputTensor.GetTensorMemoryInfo();
    Ort::Allocator allocator(session, memoryInfo);
    
    // Get the inputResource and populate!
    ComPtr<ID3D12Resource> inputResource;
    Ort::ThrowOnError(ortDmlApi->GetD3D12ResourceFromAllocation(allocator, inputTensor.GetTensorMutableData<void*>(), &inputResource));

    // Create output tensor
    type_info = session.GetOutputTypeInfo(0);
    tensor_info = type_info.GetTensorTypeAndShapeInfo();
    auto output = CreateDmlValue(tensor_info, commandQueue.Get());
    auto outputTensor = std::move(output.first);

    // Run warmup
    auto startWarm = std::chrono::system_clock::now();
    session.Run(Ort::RunOptions{ nullptr }, &inputName, &inputTensor, 1, &outputName, &outputTensor, 1);

    // Queue fence, and wait for completion
    ComPtr<ID3D12Fence> fence;
    d3dDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.GetAddressOf()));
    commandQueue->Signal(fence.Get(), 1);

    wil::unique_handle fenceEvent(CreateEvent(nullptr, FALSE, FALSE, nullptr));
    fence->SetEventOnCompletion(1, fenceEvent.get());
    WaitForSingleObject(fenceEvent.get(), INFINITE);

    auto endWarm = std::chrono::system_clock::now();
    printf("Warm Took: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(endWarm - startWarm).count() / 1000.f);

    // Record start
    auto start = std::chrono::system_clock::now();

    // Run performance test
    for (int i = fenceValueStart; i < (numIterations + fenceValueStart); i++)
    {
        session.Run(Ort::RunOptions{ nullptr }, &inputName, &inputTensor, 1, &outputName, &outputTensor, 1);

        {
            // Synchronize with CPU before queuing more inference runs
            commandQueue->Signal(fence.Get(), i);
            ResetEvent(fenceEvent.get());
            fence->SetEventOnCompletion(i, fenceEvent.get());
            WaitForSingleObject(fenceEvent.get(), INFINITE);
        }
    }

    // Record end and calculate duration
    auto end = std::chrono::system_clock::now();;
    float duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.f;
    printf("Average inference time: %f ms\n", duration / numIterations);

    // Read results
    ComPtr<ID3D12Resource> outputResource;
    Ort::ThrowOnError(ortDmlApi->GetD3D12ResourceFromAllocation(allocator, outputTensor.GetTensorMutableData<void*>(), &outputResource));
}
