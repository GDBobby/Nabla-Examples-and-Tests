#include "nbl/application_templates/BasicMultiQueueApplication.hpp"
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"

//#include "app_resources/descriptors.hlsl"
#include <nbl/builtin/hlsl/central_limit_blur/common.hlsl>


#include "nbl/builtin/CArchive.h"

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;

constexpr uint32_t WorkgroupSize = 256;
constexpr uint32_t PassesPerAxis = 4;

class BoxBlurDemo final
    : public application_templates::BasicMultiQueueApplication,
    public application_templates::
    MonoAssetManagerAndBuiltinResourceApplication {
    using base_t = application_templates::BasicMultiQueueApplication;
    using asset_base_t =
        application_templates::MonoAssetManagerAndBuiltinResourceApplication;

public:
    BoxBlurDemo(const path& _localInputCWD, const path& _localOutputCWD,
        const path& _sharedInputCWD, const path& _sharedOutputCWD)
        : system::IApplicationFramework(_localInputCWD, _localOutputCWD,
            _sharedInputCWD, _sharedOutputCWD) {}

    bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override {
        // Remember to call the base class initialization!
        if (!base_t::onAppInitialized(core::smart_refctd_ptr(system))) {
            return false;
        }
        if (!asset_base_t::onAppInitialized(std::move(system))) {
            return false;
        }

        m_imageLoadedSemaphore = m_device->createSemaphore(0);
        loadImage();
        const auto& gpuImgParams = m_image->getCreationParameters();

        video::IGPUImage::SCreationParams outImageCreateInfo;
        outImageCreateInfo.flags = {};
        outImageCreateInfo.type = gpuImgParams.type;
        outImageCreateInfo.extent = gpuImgParams.extent;
        outImageCreateInfo.mipLevels = gpuImgParams.mipLevels;
        outImageCreateInfo.arrayLayers = gpuImgParams.arrayLayers;
        outImageCreateInfo.samples = gpuImgParams.samples;
        outImageCreateInfo.tiling = video::IGPUImage::TILING::OPTIMAL;
        outImageCreateInfo.usage = gpuImgParams.usage | IImage::EUF_STORAGE_BIT | IImage::EUF_TRANSFER_DST_BIT;
        outImageCreateInfo.queueFamilyIndexCount = 0u;
        outImageCreateInfo.queueFamilyIndices = nullptr;
        outImageCreateInfo.format = E_FORMAT::EF_R8G8B8A8_UNORM;
        auto outImage = m_device->createImage(std::move(outImageCreateInfo));

        auto outImageMemReqs = outImage->getMemoryReqs();
        outImageMemReqs.memoryTypeBits &= m_physicalDevice->getDeviceLocalMemoryTypeBits();
        m_device->allocate(outImageMemReqs, outImage.get(), video::IDeviceMemoryAllocation::EMAF_NONE);
        const auto& outImgParams = outImage->getCreationParameters();

        smart_refctd_ptr<nbl::video::IGPUImageView> sampledView;
        smart_refctd_ptr<nbl::video::IGPUImageView> unormView;
        {
            sampledView = m_device->createImageView({
                .flags = IGPUImageView::ECF_NONE,
                .subUsages = IImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT,
                .image = m_image,
                .viewType = IGPUImageView::ET_2D,
                .format = gpuImgParams.format,
                });
            sampledView->setObjectDebugName("Sampled sRGB view");

            unormView = m_device->createImageView({
                .flags = IGPUImageView::ECF_NONE,
                .subUsages = IImage::E_USAGE_FLAGS::EUF_STORAGE_BIT,
                .image = outImage,
                .viewType = IGPUImageView::ET_2D,
                .format = outImgParams.format,
                });
            unormView->setObjectDebugName("UNORM view");
        }
        assert(m_image && outImage && sampledView && unormView);

        smart_refctd_ptr<IGPUShader> shader;
        {
            IAssetLoader::SAssetLoadParams lp;
            lp.logger = m_logger.get();

            SAssetBundle bundle = m_assetMgr->getAsset("../app_resources/main.comp.hlsl", lp);
            if (bundle.getContents().empty()) {
                m_logger->log("Couldn't load an asset.", ILogger::ELL_ERROR);
                std::exit(-1);
            }

            auto computeMain = IAsset::castDown<ICPUShader>(bundle.getContents()[0]);
            if (!computeMain) {
                m_logger->log("Failed to load shader", ILogger::ELL_ERROR);
                std::exit(-1);
            }
            shader = m_device->createShader(computeMain.get());
            if (!shader) {
                return logFail(
                    "Creation of a GPU Shader to from CPU Shader source failed!");
            }
        }

        smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout;
        {
            NBL_CONSTEXPR_STATIC nbl::video::IGPUDescriptorSetLayout::SBinding
                bindings[] = {
                    {.binding = 0, // TODO: ALI
                     .type =
                         nbl::asset::IDescriptor::E_TYPE::ET_SAMPLED_IMAGE,
                     .createFlags =
                         IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
                     .stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
                     .count = 1,
                     .immutableSamplers = nullptr},
                    {.binding = 1, // TODO: ALI
                     .type = nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
                     .createFlags =
                         IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
                     .stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
                     .count = 1,
                     .immutableSamplers = nullptr} };
            dsLayout = m_device->createDescriptorSetLayout(bindings);
            if (!dsLayout) {
                return logFail("Failed to create a Descriptor Layout!\n");
            }
        }

        const asset::SPushConstantRange pushConst[] = {
            {.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
             .offset = 0,
             .size = sizeof(nbl::hlsl::central_limit_blur::BoxBlurParams)} };
        smart_refctd_ptr<nbl::video::IGPUPipelineLayout> pplnLayout =
            m_device->createPipelineLayout(pushConst, smart_refctd_ptr(dsLayout));
        if (!pplnLayout) {
            return logFail("Failed to create a Pipeline Layout!\n");
        }

        smart_refctd_ptr<nbl::video::IGPUComputePipeline> pipeline;
        {
            IGPUComputePipeline::SCreationParams params = {};
            params.layout = pplnLayout.get();
            params.shader.entryPoint = "main";
            params.shader.shader = shader.get();
            if (!m_device->createComputePipelines(nullptr, { &params, 1 }, &pipeline)) {
                return logFail(
                    "Failed to create pipelines (compile & link shaders)!\n");
            }
        }
        smart_refctd_ptr<nbl::video::IDescriptorPool> pool =
            m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE,
                { &dsLayout.get(), 1 });
        smart_refctd_ptr<nbl::video::IGPUDescriptorSet> ds =
            pool->createDescriptorSet(std::move(dsLayout));
        {
            // Views must be in the same layout because we read from them
            // simultaneously
            smart_refctd_ptr<video::IGPUSampler> sampler = m_device->createSampler({});
            IGPUDescriptorSet::SDescriptorInfo info[2];
            info[0].desc = sampledView;
            info[0].info.combinedImageSampler.sampler = sampler;
            info[0].info.combinedImageSampler.imageLayout = IImage::LAYOUT::GENERAL;
            info[1].desc = unormView;
            info[1].info.image.imageLayout = IImage::LAYOUT::GENERAL;

            IGPUDescriptorSet::SWriteDescriptorSet writes[] = {
                {.dstSet = ds.get(),
                 .binding = 0, // TODO
                 .arrayElement = 0,
                 .count = 1,
                 .info = &info[0]},
                {.dstSet = ds.get(),
                 .binding = 1, // TODO
                 .arrayElement = 0,
                 .count = 1,
                 .info = &info[1]},
            };
            const bool success = m_device->updateDescriptorSets(writes, {});
            assert(success);
        }

        ds->setObjectDebugName("Box blur DS");
        pplnLayout->setObjectDebugName("Box Blur PPLN Layout");

        IQueue* queue = getComputeQueue();

        IImage::SSubresourceLayers subresourceLayers;
        subresourceLayers.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
        subresourceLayers.mipLevel = 0u;
        subresourceLayers.baseArrayLayer = 0u;
        subresourceLayers.layerCount = 1u;

        IImage::SBufferCopy bufferCopy;
        bufferCopy.bufferImageHeight = outImgParams.extent.height;
        bufferCopy.bufferRowLength = outImgParams.extent.width;
        bufferCopy.bufferOffset = 0u;
        bufferCopy.imageExtent = outImgParams.extent;
        bufferCopy.imageSubresource = subresourceLayers;

        nbl::video::IDeviceMemoryAllocator::SAllocation outputBufferAllocation = {};
        smart_refctd_ptr<IGPUBuffer> outputImageBuffer = nullptr;
        {
            IGPUBuffer::SCreationParams gpuBufCreationParams;
            gpuBufCreationParams.size = outImage->getImageDataSizeInBytes();
            gpuBufCreationParams.usage =
                IGPUBuffer::E_USAGE_FLAGS::EUF_TRANSFER_DST_BIT;
            outputImageBuffer =
                m_device->createBuffer(std::move(gpuBufCreationParams));
            if (!outputImageBuffer)
                return logFail("Failed to create a GPU Buffer of size %d!\n",
                    gpuBufCreationParams.size);

            // Naming objects is cool because not only errors (such as Vulkan
            // Validation Layers) will show their names, but RenderDoc captures too.
            outputImageBuffer->setObjectDebugName("Output Image Buffer");

            nbl::video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs =
                outputImageBuffer->getMemoryReqs();
            // you can simply constrain the memory requirements by AND-ing the type
            // bits of the host visible memory types
            reqs.memoryTypeBits &= m_physicalDevice->getHostVisibleMemoryTypeBits();

            outputBufferAllocation =
                m_device->allocate(reqs, outputImageBuffer.get(),
                    nbl::video::IDeviceMemoryAllocation::
                    E_MEMORY_ALLOCATE_FLAGS::EMAF_NONE);
            if (!outputBufferAllocation.isValid())
                return logFail("Failed to allocate Device Memory compatible with our "
                    "GPU Buffer!\n");
        }

        constexpr size_t StartedValue = 0;
        constexpr size_t FinishedValue = 45;
        static_assert(StartedValue < FinishedValue);
        smart_refctd_ptr<ISemaphore> progress =
            m_device->createSemaphore(StartedValue);

        smart_refctd_ptr<nbl::video::IGPUCommandBuffer> cmdbuf;
        smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool =
            m_device->createCommandPool(
                queue->getFamilyIndex(),
                IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
        if (!cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,
            1u, &cmdbuf)) {
            return logFail("Failed to create Command Buffers!\n");
        }

        hlsl::central_limit_blur::BoxBlurParams pushConstData = {
          .flip = 0,
          .filterDim = 15,
          .blockDim = 128 - (15 - 1),
        };

        cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
        cmdbuf->beginDebugMarker("Box Blur dispatches",
            core::vectorSIMDf(0, 1, 0, 1));

        const IGPUCommandBuffer::SImageMemoryBarrier<
            IGPUCommandBuffer::SOwnershipTransferBarrier>
            barriers[] = { {
                .barrier =
                    {
                        .dep =
                            {
                                .srcStageMask =
                                    nbl::asset::PIPELINE_STAGE_FLAGS::COPY_BIT,
                                .srcAccessMask =
                                    nbl::asset::ACCESS_FLAGS::TRANSFER_WRITE_BIT,
                                .dstStageMask = nbl::asset::PIPELINE_STAGE_FLAGS::
                                                    COMPUTE_SHADER_BIT |
                                                nbl::asset::PIPELINE_STAGE_FLAGS::
                                                    ALL_TRANSFER_BITS,
                                .dstAccessMask =
                                    nbl::asset::ACCESS_FLAGS::STORAGE_WRITE_BIT,
                            },
                    },
                .image = m_image.get(),
                .subresourceRange = {.aspectMask = IImage::EAF_COLOR_BIT,
                                     .levelCount = 1,
                                     .layerCount = 1},
                .oldLayout = IImage::LAYOUT::UNDEFINED,
                .newLayout = IImage::LAYOUT::GENERAL,
            },
            {
                .barrier =
                    {
                        .dep =
                            {
                                .srcStageMask =
                                    nbl::asset::PIPELINE_STAGE_FLAGS::COPY_BIT,
                                .srcAccessMask =
                                    nbl::asset::ACCESS_FLAGS::TRANSFER_WRITE_BIT,
                                .dstStageMask = nbl::asset::PIPELINE_STAGE_FLAGS::
                                                    COMPUTE_SHADER_BIT |
                                                nbl::asset::PIPELINE_STAGE_FLAGS::
                                                    ALL_TRANSFER_BITS,
                                .dstAccessMask =
                                    nbl::asset::ACCESS_FLAGS::STORAGE_WRITE_BIT,
                            },
                    },
                .image = outImage.get(),
                .subresourceRange = {.aspectMask = IImage::EAF_COLOR_BIT,
                                     .levelCount = 1,
                                     .layerCount = 1},
                .oldLayout = IImage::LAYOUT::UNDEFINED,
                .newLayout = IImage::LAYOUT::GENERAL,
            } };
        if (!cmdbuf->pipelineBarrier(nbl::asset::EDF_NONE,
            { .imgBarriers = barriers }))
            return logFail("Failed to issue barrier!\n");

        cmdbuf->bindComputePipeline(pipeline.get());
        cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, pplnLayout.get(), 0, 1,
            &ds.get());
        cmdbuf->pushConstants(pplnLayout.get(),
            IShader::E_SHADER_STAGE::ESS_COMPUTE, 0,
            sizeof(pushConstData), &pushConstData);

        for (int j = 0; j < 1; j++) {
            cmdbuf->dispatch(
                outImgParams.extent.width / (float)pushConstData.blockDim,
                outImgParams.extent.height / 4, // TODO: 4 = batch[1]
                1
            );
            pushConstData.flip = 1;
            cmdbuf->pushConstants(pplnLayout.get(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0, sizeof(pushConstData), &pushConstData);
            cmdbuf->dispatch(
                outImgParams.extent.height / (float)pushConstData.blockDim,
                outImgParams.extent.width / 4, // TODO: 4 = batch[1]
                1
            );

            // const nbl::asset::SMemoryBarrier barriers3[] = {
            // 	{
            // 		.srcStageMask =
            // nbl::asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT, 		.srcAccessMask =
            // nbl::asset::ACCESS_FLAGS::SHADER_WRITE_BITS, 		.dstStageMask =
            // nbl::asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT, 		.dstAccessMask=
            // nbl::asset::ACCESS_FLAGS::SHADER_READ_BITS,
            // 	}
            // };
            // // TODO: you don't need a pipeline barrier just before the end of the
            // last command buffer to be submitted
            // // Timeline semaphore takes care of all the memory deps between a
            // signal and a wait if( !cmdbuf->pipelineBarrier( nbl::asset::EDF_NONE, {
            // .memBarriers = barriers3 } ) )
            // {
            // 	return logFail( "Failed to issue barrier!\n" );
            // }

            // pushConstData.direction = 1;
            // cmdbuf->pushConstants( pplnLayout.get(),
            // IShader::E_SHADER_STAGE::ESS_COMPUTE, 0, sizeof( pushConstData ),
            // &pushConstData ); cmdbuf->dispatch(1, outImgParams.extent.width, 1);
        }

        const IGPUCommandBuffer::SImageMemoryBarrier<
            IGPUCommandBuffer::SOwnershipTransferBarrier>
            barriers2[] = { {
                .barrier =
                    {
                        .dep =
                            {
                                .srcStageMask = nbl::asset::PIPELINE_STAGE_FLAGS::
                                    COMPUTE_SHADER_BIT,
                                .srcAccessMask =
                                    nbl::asset::ACCESS_FLAGS::STORAGE_WRITE_BIT,
                                .dstStageMask = nbl::asset::PIPELINE_STAGE_FLAGS::
                                    ALL_TRANSFER_BITS,
                                .dstAccessMask =
                                    nbl::asset::ACCESS_FLAGS::MEMORY_READ_BITS,
                            },
                    },
                .image = outImage.get(),
                .subresourceRange = {.aspectMask = IImage::EAF_COLOR_BIT,
                                     .levelCount = 1,
                                     .layerCount = 1},
                .oldLayout = IImage::LAYOUT::UNDEFINED,
                .newLayout = IImage::LAYOUT::TRANSFER_SRC_OPTIMAL,
            } };
        if (!cmdbuf->pipelineBarrier(nbl::asset::EDF_NONE,
            { .imgBarriers = barriers2 }))
            return logFail("Failed to issue barrier!\n");

        // Copy the resulting image to a buffer.
        cmdbuf->copyImageToBuffer(outImage.get(),
            IImage::LAYOUT::TRANSFER_SRC_OPTIMAL,
            outputImageBuffer.get(), 1u, &bufferCopy);

        cmdbuf->endDebugMarker();
        cmdbuf->end();

        {
            const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[] = {
                {.cmdbuf = cmdbuf.get()} };
            const IQueue::SSubmitInfo::SSemaphoreInfo waits[] = {
                {.semaphore = m_imageLoadedSemaphore.get(),
                 .value = 1,
                 .stageMask = asset::PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS} };
            const IQueue::SSubmitInfo::SSemaphoreInfo signals[] = {
                {.semaphore = progress.get(),
                 .value = FinishedValue,
                 .stageMask = asset::PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS} };
            IQueue::SSubmitInfo submitInfos[] = {
                {.waitSemaphores = waits,
                 .commandBuffers = cmdbufs,
                 .signalSemaphores = signals} };

            // This is super useful for debugging multi-queue workloads and by default
            // RenderDoc delimits captures only by Swapchain presents.
//            queue->startCapture();
            queue->submit(submitInfos);
 //           queue->endCapture();
        }
        const ISemaphore::SWaitInfo waitInfos[] = {
            {.semaphore = progress.get(), .value = FinishedValue} };
        m_device->blockForSemaphores(waitInfos);

        // Map memory, so contents of `outputImageBuffer` will be host visible.
        const ILogicalDevice::MappedMemoryRange memoryRange(
            outputBufferAllocation.memory.get(), 0ull,
            outputBufferAllocation.memory->getAllocationSize());
        auto imageBufferMemPtr = outputBufferAllocation.memory->map(
            { 0ull, outputBufferAllocation.memory->getAllocationSize() },
            IDeviceMemoryAllocation::EMCAF_READ);
        if (!imageBufferMemPtr)
            return logFail("Failed to map the Device Memory!\n");

        // If the mapping is not coherent the range needs to be invalidated to pull
        // in new data for the CPU's caches.
        if (!outputBufferAllocation.memory->getMemoryPropertyFlags().hasFlags(
            IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
            m_device->invalidateMappedMemoryRanges(1, &memoryRange);

        // While JPG/PNG/BMP/EXR Loaders create ICPUImages because they cannot
        // disambiguate colorspaces, 2D_ARRAY vs 2D and even sometimes formats
        // (looking at your PNG normalmaps!), the writers are always meant to be fed
        // by ICPUImageViews.
        ICPUImageView::SCreationParams params = {};
        {

            // ICPUImage isn't really a representation of a GPU Image in itself, more
            // of a recipe for creating one from a series of ICPUBuffer to ICPUImage
            // copies. This means that an ICPUImage has no internal storage or memory
            // bound for its texels and rather references separate ICPUBuffer ranges
            // to provide its contents, which also means it can be sparsely(with gaps)
            // specified.
            params.image = ICPUImage::create(outImgParams);
            {
                // CDummyCPUBuffer is used for creating ICPUBuffer over an already
                // existing memory, without any memcopy operations or taking over the
                // memory ownership. CDummyCPUBuffer cannot free its memory.
                auto cpuOutputImageBuffer =
                    core::make_smart_refctd_ptr<CDummyCPUBuffer>(
                        outImage->getImageDataSizeInBytes(), imageBufferMemPtr,
                        core::adopt_memory_t());
                ICPUImage::SBufferCopy region = {};
                region.imageSubresource.aspectMask =
                    IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
                region.imageSubresource.layerCount = 1;
                region.imageExtent = outImgParams.extent;

                //
                params.image->setBufferAndRegions(
                    std::move(cpuOutputImageBuffer),
                    core::make_refctd_dynamic_array<
                    core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(
                        1, region));
            }
            // Only DDS and KTX support exporting layered views.
            params.viewType = ICPUImageView::ET_2D;
            params.format = outImgParams.format;
            params.subresourceRange.aspectMask =
                IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
            params.subresourceRange.layerCount = 1;
        }
        auto cpuImageView = ICPUImageView::create(std::move(params));
        asset::IAssetWriter::SAssetWriteParams writeParams(cpuImageView.get());
        m_assetMgr->writeAsset("blit.png", writeParams);

        // Even if you forgot to unmap, it would unmap itself when
        // `outputBufferAllocation.memory` gets dropped by its last reference and
        // its destructor runs.
        outputBufferAllocation.memory->unmap();

        return true;
    }

    void loadImage()
    {
        IAssetLoader::SAssetLoadParams lp;
        lp.logger = m_logger.get();

        auto transferUpQueue = getTransferUpQueue();

        // intialize command buffers
        core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer> commandBuffer;
        m_device->createCommandPool(
            transferUpQueue->getFamilyIndex(),
            IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT
        )->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1, &commandBuffer, core::smart_refctd_ptr(m_logger));
        //
        IQueue::SSubmitInfo::SCommandBufferInfo commandBufferInfo;
            commandBuffer->setObjectDebugName("Upload Command Buffer");
            commandBufferInfo.cmdbuf = commandBuffer.get();

        core::smart_refctd_ptr<ISemaphore> imgFillSemaphore = m_device->createSemaphore(0);
        imgFillSemaphore->setObjectDebugName("Image Fill Semaphore");

        //
        auto converter = CAssetConverter::create({ .device = m_device.get() });
        // We don't want to generate mip-maps for these images, to ensure that we must override the default callbacks.
        struct SInputs final : CAssetConverter::SInputs
        {
            // we also need to override this to have concurrent sharing
            inline std::span<const uint32_t> getSharedOwnershipQueueFamilies(const size_t groupCopyID, const asset::ICPUImage* buffer, const CAssetConverter::patch_t<asset::ICPUImage>& patch) const override
            {
                if (familyIndices.size() > 1)
                    return familyIndices;
                return {};
            }

            inline uint8_t getMipLevelCount(const size_t groupCopyID, const ICPUImage* image, const CAssetConverter::patch_t<asset::ICPUImage>& patch) const override
            {
                return image->getCreationParameters().mipLevels;
            }
            inline uint16_t needToRecomputeMips(const size_t groupCopyID, const ICPUImage* image, const CAssetConverter::patch_t<asset::ICPUImage>& patch) const override
            {
                return 0b0u;
            }

            std::vector<uint32_t> familyIndices;
        } inputs = {};
        inputs.readCache = converter.get();
        inputs.logger = m_logger.get();
        {
            const core::set<uint32_t> uniqueFamilyIndices = { getTransferUpQueue()->getFamilyIndex(), getComputeQueue()->getFamilyIndex() };
            inputs.familyIndices = { uniqueFamilyIndices.begin(),uniqueFamilyIndices.end() };
        }
        // scratch command buffers for asset converter transfer commands
        SIntendedSubmitInfo transfer = {
            .queue = transferUpQueue,
            .waitSemaphores = {},
            .prevCommandBuffers = {},
            .scratchCommandBuffers = { &commandBufferInfo, 1 },
            .scratchSemaphore = {
                .semaphore = imgFillSemaphore.get(),
                .value = 0,
                // because of layout transitions
                .stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS
            }
        };
        // as per the `SIntendedSubmitInfo` one commandbuffer must be begun
        commandBuffer->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
        // Normally we'd have to inherit and override the `getFinalOwnerQueueFamily` callback to ensure that the
        // compute queue becomes the owner of the buffers and images post-transfer, but in this example we use concurrent sharing
        CAssetConverter::SConvertParams params = {};
        params.transfer = &transfer;
        params.utilities = m_utils.get();

        const auto imagePathToLoad = "../app_resources/tex.jpg";

        SAssetBundle bundle = m_assetMgr->getAsset(imagePathToLoad, lp);
        if (bundle.getContents().empty()) {
            m_logger->log("Couldn't load an asset.", ILogger::ELL_ERROR);
            std::exit(-1);
        }

        auto cpuImage = IAsset::castDown<ICPUImage>(bundle.getContents()[0]);
        cpuImage->addImageUsageFlags(ICPUImage::E_USAGE_FLAGS::EUF_STORAGE_BIT | ICPUImage::E_USAGE_FLAGS::EUF_TRANSFER_SRC_BIT);
        if (!cpuImage) {
            m_logger->log("Failed to load image from path %s", ILogger::ELL_ERROR, imagePathToLoad);
            std::exit(-1);
        }

        std::get<CAssetConverter::SInputs::asset_span_t<ICPUImage>>(inputs.assets) = { &cpuImage.get(),1 };
        // assert that we don't need to provide patches
        assert(cpuImage->getImageUsageFlags().hasFlags(ICPUImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT));
        auto reservation = converter->reserve(inputs);
        // the `.value` is just a funny way to make the `smart_refctd_ptr` copyable
        m_image = reservation.getGPUObjects<ICPUImage>().front().value;
        if (!m_image) {
            m_logger->log("Failed to convert %s into an IGPUImage handle", ILogger::ELL_ERROR, imagePathToLoad);
            std::exit(-1);
        }

        // debug log about overflows
        transfer.overflowCallback = [&](const ISemaphore::SWaitInfo&)->void
            {
                m_logger->log("Overflown when uploading image nr!\n", ILogger::ELL_PERFORMANCE);
            };
        // we want our converter's submit to signal a semaphore that image contents are ready
        const IQueue::SSubmitInfo::SSemaphoreInfo signalSemaphore = {
                .semaphore = m_imageLoadedSemaphore.get(),
                .value = 1u,
                // cannot signal from COPY stage because there's a layout transition and a possible ownership transfer
                // and we need to wait for right after and they don't have an explicit stage
                .stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS
        };
        params.extraSignalSemaphores = { &signalSemaphore,1 };
        // and launch the conversions
        transferUpQueue->startCapture();
        auto result = reservation.convert(params);
        transferUpQueue->endCapture();
        if (!result.blocking() && result.copy() != IQueue::RESULT::SUCCESS) {
            m_logger->log("Failed to record or submit conversions");
            std::exit(-1);
        }

    }

    // Platforms like WASM expect the main entry point to periodically return
    // control, hence if you want a crossplatform app, you have to let the
    // framework deal with your "game loop"
    void workLoopBody() override {}

    // Whether to keep invoking the above. In this example because its headless
    // GPU compute, we do all the work in the app initialization.
    bool keepRunning() override { return false; }

    // Just to run destructors in a nice order
    bool onAppTerminated() override { return base_t::onAppTerminated(); }

private:
    smart_refctd_ptr<ISemaphore> m_imageLoadedSemaphore;
    core::smart_refctd_ptr<IGPUImage> m_image;
};

NBL_MAIN_FUNC(BoxBlurDemo)