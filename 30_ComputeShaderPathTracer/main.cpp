// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/this_example/common.hpp"
#include "nbl/asset/interchange/IImageAssetHandlerBase.h"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"


using namespace nbl;
using namespace core;
using namespace hlsl;
using namespace system;
using namespace asset;
using namespace ui;
using namespace video;

// TODO: Add a QueryPool for timestamping once its ready
// TODO: Do buffer creation using assConv
class ComputeShaderPathtracer final : public examples::SimpleWindowedApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
		using device_base_t = examples::SimpleWindowedApplication;
		using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;
		using clock_t = std::chrono::steady_clock;

		constexpr static inline uint32_t2 WindowDimensions = { 1280, 720 };
		constexpr static inline uint32_t FramesInFlight = 1;
		constexpr static inline clock_t::duration DisplayImageDuration = std::chrono::milliseconds(900);
		constexpr static inline uint32_t DefaultWorkGroupSize = 16u;
		constexpr static inline uint32_t MaxDescriptorCount = 256u;
		constexpr static inline uint32_t MaxDepthLog2 = 4u; // 5
		constexpr static inline uint32_t MaxSamplesLog2 = 10u; // 18
		constexpr static inline uint32_t MaxBufferDimensions = 3u << MaxDepthLog2;
		constexpr static inline uint32_t MaxBufferSamples = 1u << MaxSamplesLog2;
		constexpr static inline uint8_t MaxUITextureCount = 2u;
		constexpr static inline uint8_t SceneTextureIndex = 1u;
		static inline std::string DefaultImagePathsFile = "../../media/envmap/envmap_0.exr";
		static inline std::array<std::string, 3> PTShaderPaths = { "app_resources/litBySphere.comp", "app_resources/litByTriangle.comp", "app_resources/litByRectangle.comp" };
		static inline std::string PresentShaderPath = "app_resources/present.frag.hlsl";

	public:
		inline ComputeShaderPathtracer(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
			: IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {
			const auto cameraPos = core::vectorSIMDf(0, 5, -10);
			matrix4SIMD proj = matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(
				core::radians(fov),
				static_cast<float32_t>(WindowDimensions.x) / static_cast<float32_t>(WindowDimensions.y),
				zNear,
				zFar
			);

			m_camera = Camera(cameraPos, core::vectorSIMDf(0, 0, 0), proj);
		}

		inline bool isComputeOnly() const override { return false; }

		inline core::vector<video::SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const override
		{
			if (!m_surface)
			{
				{
					auto windowCallback = core::make_smart_refctd_ptr<CEventCallback>(smart_refctd_ptr(m_inputSystem), smart_refctd_ptr(m_logger));
					IWindow::SCreationParams params = {};
					params.callback = core::make_smart_refctd_ptr<nbl::video::ISimpleManagedSurface::ICallback>();
					params.width = WindowDimensions.x;
					params.height = WindowDimensions.y;
					params.x = 32;
					params.y = 32;
					params.flags = ui::IWindow::ECF_HIDDEN | IWindow::ECF_BORDERLESS | IWindow::ECF_RESIZABLE;
					params.windowCaption = "ComputeShaderPathtracer";
					params.callback = windowCallback;
					const_cast<std::remove_const_t<decltype(m_window)>&>(m_window) = m_winMgr->createWindow(std::move(params));
				}

				auto surface = CSurfaceVulkanWin32::create(smart_refctd_ptr(m_api), smart_refctd_ptr_static_cast<IWindowWin32>(m_window));
				const_cast<std::remove_const_t<decltype(m_surface)>&>(m_surface) = nbl::video::CSimpleResizeSurface<nbl::video::CDefaultSwapchainFramebuffers>::create(std::move(surface));
			}

			if (m_surface)
				return { {m_surface->getSurface()/*,EQF_NONE*/} };

			return {};
		}

		inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			// Init systems
			{
				m_inputSystem = make_smart_refctd_ptr<InputSystem>(logger_opt_smart_ptr(smart_refctd_ptr(m_logger)));

				// Remember to call the base class initialization!
				if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
					return false;
				if (!asset_base_t::onAppInitialized(std::move(system)))
					return false;

				m_semaphore = m_device->createSemaphore(m_realFrameIx);
				if (!m_semaphore)
					return logFail("Failed to create semaphore!");
			}

			// Create renderpass and init surface
			nbl::video::IGPURenderpass* renderpass;
			{
				ISwapchain::SCreationParams swapchainParams = { .surface = smart_refctd_ptr<ISurface>(m_surface->getSurface()) };
				if (!swapchainParams.deduceFormat(m_physicalDevice))
					return logFail("Could not choose a Surface Format for the Swapchain!");

				const static IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] =
				{
					{
						.srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
						.dstSubpass = 0,
						.memoryBarrier =
						{
							.srcStageMask = asset::PIPELINE_STAGE_FLAGS::COPY_BIT,
							.srcAccessMask = asset::ACCESS_FLAGS::TRANSFER_WRITE_BIT,
							.dstStageMask = asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
							.dstAccessMask = asset::ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
						}
					},
					{
						.srcSubpass = 0,
						.dstSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
						.memoryBarrier =
						{
							.srcStageMask = asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
							.srcAccessMask = asset::ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
						}
					},
					IGPURenderpass::SCreationParams::DependenciesEnd
				};

				auto scResources = std::make_unique<CDefaultSwapchainFramebuffers>(m_device.get(), swapchainParams.surfaceFormat.format, dependencies);
				renderpass = scResources->getRenderpass();

				if (!renderpass)
					return logFail("Failed to create Renderpass!");

				auto gQueue = getGraphicsQueue();
				if (!m_surface || !m_surface->init(gQueue, std::move(scResources), swapchainParams.sharedParams))
					return logFail("Could not create Window & Surface or initialize the Surface!");
			}

			// Create command pool and buffers
			{
				auto gQueue = getGraphicsQueue();
				m_cmdPool = m_device->createCommandPool(gQueue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
				if (!m_cmdPool)
					return logFail("Couldn't create Command Pool!");

				if (!m_cmdPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { m_cmdBufs.data(), 2 }))
					return logFail("Couldn't create Command Buffer!");
			}

			ISampler::SParams samplerParams = {
				.AnisotropicFilter = 0
			};
			auto cpuSampler = make_smart_refctd_ptr<ICPUSampler>(samplerParams);
			auto gpuSampler = m_device->createSampler(std::move(samplerParams));

			// Create descriptors and pipeline for the pathtracer
			{
				auto convertDSLayoutCPU2GPU = [&](smart_refctd_ptr<ICPUDescriptorSetLayout> cpuLayout) {
					auto converter = CAssetConverter::create({ .device = m_device.get() });
					CAssetConverter::SInputs inputs = {};
					inputs.readCache = converter.get();
					inputs.logger = m_logger.get();
					CAssetConverter::SConvertParams params = {};
					params.utilities = m_utils.get();

					std::get<CAssetConverter::SInputs::asset_span_t<ICPUDescriptorSetLayout>>(inputs.assets) = { &cpuLayout.get(),1 };
					// don't need to assert that we don't need to provide patches since layouts are not patchable
					//assert(true);
					auto reservation = converter->reserve(inputs);
					// the `.value` is just a funny way to make the `smart_refctd_ptr` copyable
					auto gpuLayout = reservation.getGPUObjects<ICPUDescriptorSetLayout>().front().value;
					if (!gpuLayout) {
						m_logger->log("Failed to convert %s into an IGPUDescriptorSetLayout handle", ILogger::ELL_ERROR);
						std::exit(-1);
					}

					return gpuLayout;
					};
				auto convertDSCPU2GPU = [&](smart_refctd_ptr<ICPUDescriptorSet> cpuDS) {
					auto converter = CAssetConverter::create({ .device = m_device.get() });
					CAssetConverter::SInputs inputs = {};
					inputs.readCache = converter.get();
					inputs.logger = m_logger.get();
					CAssetConverter::SConvertParams params = {};
					params.utilities = m_utils.get();

					std::get<CAssetConverter::SInputs::asset_span_t<ICPUDescriptorSet>>(inputs.assets) = { &cpuDS.get(), 1 };
					// don't need to assert that we don't need to provide patches since layouts are not patchable
					//assert(true);
					auto reservation = converter->reserve(inputs);
					// the `.value` is just a funny way to make the `smart_refctd_ptr` copyable
					auto gpuDS = reservation.getGPUObjects<ICPUDescriptorSet>().front().value;
					if (!gpuDS) {
						m_logger->log("Failed to convert %s into an IGPUDescriptorSet handle", ILogger::ELL_ERROR);
						std::exit(-1);
					}

					return gpuDS;
					};

				std::array<ICPUDescriptorSetLayout::SBinding, 1> cpuDSBinding = {};
				cpuDSBinding[0] = {
					.binding = 0u,
					.type = nbl::asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
					.createFlags = ICPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = IShader::E_SHADER_STAGE::ESS_FRAGMENT,
					.count = 1u,
					.immutableSamplers = &cpuSampler
				};
				std::array<IGPUDescriptorSetLayout::SBinding, 1> gpuDSBinding = {};
				gpuDSBinding[0] = {
					.binding = 0u,
					.type = nbl::asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
					.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = IShader::E_SHADER_STAGE::ESS_FRAGMENT,
					.count = 1u,
					.immutableSamplers = &gpuSampler
				};

				auto cpuDSLayout = make_smart_refctd_ptr<ICPUDescriptorSetLayout>(cpuDSBinding);
				gpuDSLayout = convertDSLayoutCPU2GPU(cpuDSLayout);
				gpuDSLayoutManual = m_device->createDescriptorSetLayout(gpuDSBinding);

				auto cpuDS = make_smart_refctd_ptr<ICPUDescriptorSet>(std::move(cpuDSLayout));
				gpuDS = convertDSCPU2GPU(cpuDS);

				{
					const video::IGPUDescriptorSetLayout* const layouts[] = { gpuDSLayoutManual.get() };
					const uint32_t setCounts[] = { 1u };
					dsPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::E_CREATE_FLAGS::ECF_NONE, layouts, setCounts);
				}
				gpuDSManual = dsPool->createDescriptorSet(gpuDSLayoutManual);

				// Create Shaders
				auto loadAndCompileShader = [&](std::string pathToShader) {
					IAssetLoader::SAssetLoadParams lp = {};
					auto assetBundle = m_assetMgr->getAsset(pathToShader, lp);
					const auto assets = assetBundle.getContents();
					if (assets.empty())
					{
						m_logger->log("Could not load shader: ", ILogger::ELL_ERROR, pathToShader);
						std::exit(-1);
					}

					auto source = IAsset::castDown<ICPUShader>(std::move(assets[0]));
					// The down-cast should not fail!
					assert(source);

					// this time we skip the use of the asset converter since the ICPUShader->IGPUShader path is quick and simple
					auto shader = m_device->createShader(source.get());
					if (!shader)
					{
						m_logger->log("Shader creationed failed: %s!", ILogger::ELL_ERROR, pathToShader);
						std::exit(-1);
					}

					return shader;
				};

				// Create graphics pipeline
				{
					auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
					ext::FullScreenTriangle::ProtoPipeline fsTriProtoPPln(m_assetMgr.get(), m_device.get(), m_logger.get());
					if (!fsTriProtoPPln)
						return logFail("Failed to create Full Screen Triangle protopipeline or load its vertex shader!");

					// Load Fragment Shader
					auto fragmentShader = loadAndCompileShader(PresentShaderPath);
					if (!fragmentShader)
						return logFail("Failed to Load and Compile Fragment Shader: lumaMeterShader!");

					const IGPUShader::SSpecInfo fragSpec = {
						.entryPoint = "main",
						.shader = fragmentShader.get()
					};

					auto layout = m_device->createPipelineLayout(
						{},
						core::smart_refctd_ptr(gpuDSLayout),
						nullptr,
						nullptr,
						nullptr
					);
					pipeline = fsTriProtoPPln.createPipeline(fragSpec, layout.get(), scRes->getRenderpass());
					if (!pipeline)
						return logFail("Could not create Graphics Pipeline!");

				}
				// Create manual graphics pipeline
				{
					auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
					ext::FullScreenTriangle::ProtoPipeline fsTriProtoPPln(m_assetMgr.get(), m_device.get(), m_logger.get());
					if (!fsTriProtoPPln)
						return logFail("Failed to create Full Screen Triangle protopipeline or load its vertex shader!");

					// Load Fragment Shader
					auto fragmentShader = loadAndCompileShader(PresentShaderPath);
					if (!fragmentShader)
						return logFail("Failed to Load and Compile Fragment Shader: lumaMeterShader!");

					const IGPUShader::SSpecInfo fragSpec = {
						.entryPoint = "main",
						.shader = fragmentShader.get()
					};

					auto layout = m_device->createPipelineLayout(
						{},
						core::smart_refctd_ptr(gpuDSLayoutManual),
						nullptr,
						nullptr,
						nullptr
					);
					pipelineManual = fsTriProtoPPln.createPipeline(fragSpec, layout.get(), scRes->getRenderpass());
					if (!pipelineManual)
						return logFail("Could not create Graphics Pipeline!");

				}
			}

			// Create CPUImage and convert to GPUImage
			{
				auto convertImgCPU2GPU = [&](smart_refctd_ptr<ICPUImage> cpuImg) {
					auto queue = getGraphicsQueue();
					auto cmdbuf = m_cmdBufs[0].get();
					cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::NONE);
					std::array<IQueue::SSubmitInfo::SCommandBufferInfo, 1> commandBufferInfo = { cmdbuf };
					core::smart_refctd_ptr<ISemaphore> imgFillSemaphore = m_device->createSemaphore(0);
					imgFillSemaphore->setObjectDebugName("Image Fill Semaphore");

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
						const core::set<uint32_t> uniqueFamilyIndices = { queue->getFamilyIndex(), queue->getFamilyIndex() };
						inputs.familyIndices = { uniqueFamilyIndices.begin(),uniqueFamilyIndices.end() };
					}
					// scratch command buffers for asset converter transfer commands
					SIntendedSubmitInfo transfer = {
						.queue = queue,
						.waitSemaphores = {},
						.prevCommandBuffers = {},
						.scratchCommandBuffers = commandBufferInfo,
						.scratchSemaphore = {
							.semaphore = imgFillSemaphore.get(),
							.value = 0,
							// because of layout transitions
							.stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS
						}
					};
					// as per the `SIntendedSubmitInfo` one commandbuffer must be begun
					cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
					// Normally we'd have to inherit and override the `getFinalOwnerQueueFamily` callback to ensure that the
					// compute queue becomes the owner of the buffers and images post-transfer, but in this example we use concurrent sharing
					CAssetConverter::SConvertParams params = {};
					params.transfer = &transfer;
					params.utilities = m_utils.get();

					std::get<CAssetConverter::SInputs::asset_span_t<ICPUImage>>(inputs.assets) = { &cpuImg.get(),1 };
					// assert that we don't need to provide patches
					assert(cpuImg->getImageUsageFlags().hasFlags(ICPUImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT));
					auto reservation = converter->reserve(inputs);
					// the `.value` is just a funny way to make the `smart_refctd_ptr` copyable
					auto gpuImg = reservation.getGPUObjects<ICPUImage>().front().value;
					if (!gpuImg) {
						m_logger->log("Failed to convert %s into an IGPUImage handle", ILogger::ELL_ERROR, DefaultImagePathsFile);
						std::exit(-1);
					}

					// we want our converter's submit to signal a semaphore that image contents are ready
					const IQueue::SSubmitInfo::SSemaphoreInfo signalSemaphore = {
							.semaphore = imgFillSemaphore.get(),
							.value = 1u,
							// cannot signal from COPY stage because there's a layout transition and a possible ownership transfer
							// and we need to wait for right after and they don't have an explicit stage
							.stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS
					};
					params.extraSignalSemaphores = { &signalSemaphore,1 };
					// and launch the conversions
					m_api->startCapture();
					auto result = reservation.convert(params);
					m_api->endCapture();
					if (!result.blocking() && result.copy() != IQueue::RESULT::SUCCESS) {
						m_logger->log("Failed to record or submit conversions", ILogger::ELL_ERROR);
						std::exit(-1);
					}

					return gpuImg;
				};

				smart_refctd_ptr<ICPUImage> cpuImg;
				{
					asset::ICPUImage::SCreationParams info;
					info.format = asset::E_FORMAT::EF_R32G32B32_UINT;
					info.type = asset::ICPUImage::ET_2D;
					info.extent.width = WindowDimensions.r;
					info.extent.height = WindowDimensions.g;
					info.extent.depth = 1u;
					info.mipLevels = 1u;
					info.arrayLayers = 1u;
					info.samples = asset::ICPUImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT;
					info.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);
					info.usage = asset::IImage::EUF_TRANSFER_SRC_BIT | asset::IImage::EUF_SAMPLED_BIT;

					cpuImg = ICPUImage::create(std::move(info));
					const uint32_t texelFormatByteSize = getTexelOrBlockBytesize(cpuImg->getCreationParameters().format);
					const uint32_t texelBufferSize = cpuImg->getImageDataSizeInBytes();
					auto texelBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(texelBufferSize);

					auto out = reinterpret_cast<uint8_t *>(texelBuffer->getPointer());
					for (auto index = 0u; index < texelBufferSize; index ++) {
						double alpha = static_cast<double>(index) / texelBufferSize;
						out[index] = 256 * alpha;
					}

					auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(1u);
					ICPUImage::SBufferCopy& region = regions->front();
					region.imageSubresource.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
					region.imageSubresource.mipLevel = 0u;
					region.imageSubresource.baseArrayLayer = 0u;
					region.imageSubresource.layerCount = 1u;
					region.bufferOffset = 0u;
					region.bufferRowLength = IImageAssetHandlerBase::calcPitchInBlocks(WindowDimensions.r, texelFormatByteSize);
					region.bufferImageHeight = 0u;
					region.imageOffset = { 0u, 0u, 0u };
					region.imageExtent = {WindowDimensions.r, WindowDimensions.g, 1u};

					cpuImg->setBufferAndRegions(std::move(texelBuffer), regions);
				}

				img = convertImgCPU2GPU(cpuImg);
				img->setObjectDebugName("IMG");
				imgView = [this](smart_refctd_ptr<IGPUImage> img) -> smart_refctd_ptr<IGPUImageView>
				{
					auto format = img->getCreationParameters().format;
					IGPUImageView::SCreationParams imgViewInfo;
					imgViewInfo.image = std::move(img);
					imgViewInfo.format = format;
					imgViewInfo.viewType = IGPUImageView::ET_2D;
					imgViewInfo.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
					imgViewInfo.subresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
					imgViewInfo.subresourceRange.baseArrayLayer = 0u;
					imgViewInfo.subresourceRange.baseMipLevel = 0u;
					imgViewInfo.subresourceRange.layerCount = 1u;
					imgViewInfo.subresourceRange.levelCount = 1u;

					return m_device->createImageView(std::move(imgViewInfo));
				}(img);
				imgView->setObjectDebugName("IMG VIEW");
			}

			m_winMgr->setWindowSize(m_window.get(), WindowDimensions.x, WindowDimensions.y);
			m_surface->recreateSwapchain();
			m_winMgr->show(m_window.get());

			// Update Descriptors
			{
				std::array<IGPUDescriptorSet::SDescriptorInfo, 1> writeDSInfos = {};
				writeDSInfos[0].desc = imgView;
				writeDSInfos[0].info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;

				std::array<IGPUDescriptorSet::SWriteDescriptorSet, 1> writeDescriptorSets = {};
				writeDescriptorSets[0] = {
					.dstSet = gpuDS.get(),
					.binding = 0,
					.arrayElement = 0u,
					.count = 1u,
					.info = &writeDSInfos[0]
				};

				m_device->updateDescriptorSets(writeDescriptorSets, {});
			}

			auto queue = getGraphicsQueue();
			auto& cmdbuf = m_cmdBufs[1];

			m_api->startCapture();
			
			m_currentImageAcquire = m_surface->acquireNextImage();

			cmdbuf->beginDebugMarker("ComputeShaderPathtracer IMGUI Frame");
			cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			
			asset::SViewport viewport;
			{
				viewport.minDepth = 1.f;
				viewport.maxDepth = 0.f;
				viewport.x = 0u;
				viewport.y = 0u;
				viewport.width = WindowDimensions.x;
				viewport.height = WindowDimensions.y;
			}
			cmdbuf->setViewport(0u, 1u, &viewport);

			const VkRect2D currentRenderArea =
			{
				.offset = {0,0},
				.extent = {m_window->getWidth(),m_window->getHeight()}
			};

			auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
			const IGPUCommandBuffer::SRenderpassBeginInfo info =
			{
				.framebuffer = scRes->getFramebuffer(m_currentImageAcquire.imageIndex),
				.colorClearValues = &clearColor,
				.depthStencilClearValues = nullptr,
				.renderArea = currentRenderArea
			};
			cmdbuf->beginRenderPass(info, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
			cmdbuf->bindGraphicsPipeline(pipelineManual.get());
			cmdbuf->bindDescriptorSets(EPBP_GRAPHICS, pipelineManual->getLayout(), 0, 1u, &gpuDSManual.get());
			cmdbuf->endRenderPass();
			cmdbuf->end();

			auto semaphore = m_device->createSemaphore(0);

			const IQueue::SSubmitInfo::SSemaphoreInfo rendered[] =
			{
				{
					.semaphore = semaphore.get(),
					.value = 1,
					.stageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT
				}
			};
			const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[] =
			{
				{.cmdbuf = cmdbuf.get() }
			};
			const IQueue::SSubmitInfo::SSemaphoreInfo acquired[] =
			{
				{
					.semaphore = m_currentImageAcquire.semaphore,
					.value = m_currentImageAcquire.acquireCount,
					.stageMask = PIPELINE_STAGE_FLAGS::NONE
				}
			};
			const IQueue::SSubmitInfo infos[] =
			{
				{
					.waitSemaphores = acquired,
					.commandBuffers = commandBuffers,
					.signalSemaphores = rendered
				}
			};

			m_surface->present(m_currentImageAcquire.imageIndex, rendered);

			m_api->endCapture();

			return true;
		}

		inline void workLoopBody() override {}

		inline bool keepRunning() override
		{
			if (m_surface->irrecoverable())
				return false;

			return true;
		}

		inline bool onAppTerminated() override
		{
			return device_base_t::onAppTerminated();
		}

	private:
		smart_refctd_ptr<IWindow> m_window;
		smart_refctd_ptr<CSimpleResizeSurface<CDefaultSwapchainFramebuffers>> m_surface;

		smart_refctd_ptr<IDescriptorPool> dsPool;
		smart_refctd_ptr<IGPUDescriptorSetLayout> gpuDSLayout, gpuDSLayoutManual;
		smart_refctd_ptr<IGPUDescriptorSet> gpuDS, gpuDSManual;
		smart_refctd_ptr<IGPUGraphicsPipeline> pipeline, pipelineManual;
		smart_refctd_ptr<IGPUImage> img;
		smart_refctd_ptr<IGPUImageView> imgView;

		// gpu resources
		smart_refctd_ptr<IGPUCommandPool> m_cmdPool;
		smart_refctd_ptr<IGPUComputePipeline> m_PTPipeline;
		smart_refctd_ptr<IGPUGraphicsPipeline> m_presentPipeline;
		uint64_t m_realFrameIx : 59 = 0;
		uint64_t m_maxFramesInFlight : 5;
		std::array<smart_refctd_ptr<IGPUCommandBuffer>, ISwapchain::MaxImages> m_cmdBufs;
		ISimpleManagedSurface::SAcquireResult m_currentImageAcquire = {};
		smart_refctd_ptr<IGPUDescriptorSet> m_descriptorSet0, m_uboDescriptorSet1, m_descriptorSet2, m_presentDescriptorSet;

		core::smart_refctd_ptr<IDescriptorPool> m_guiDescriptorSetPool;

		// system resources
		core::smart_refctd_ptr<InputSystem> m_inputSystem;
		InputSystem::ChannelReader<IMouseEventChannel> mouse;
		InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

		// pathtracer resources
		smart_refctd_ptr<IGPUImageView> m_envMapView, m_scrambleView;
		smart_refctd_ptr<IGPUBufferView> m_sequenceBufferView;
		smart_refctd_ptr<IGPUBuffer> m_ubo;
		smart_refctd_ptr<IGPUImageView> m_outImgView;

		// sync
		smart_refctd_ptr<ISemaphore> m_semaphore;

		// image upload resources
		smart_refctd_ptr<ISemaphore> m_scratchSemaphore;
		SIntendedSubmitInfo m_intendedSubmit;

		/*struct C_UI
		{
			nbl::core::smart_refctd_ptr<nbl::ext::imgui::UI> manager;

			struct
			{
				core::smart_refctd_ptr<video::IGPUSampler> gui, scene;
			} samplers;

			core::smart_refctd_ptr<IGPUDescriptorSet> descriptorSet;
		} m_ui; */

		Camera m_camera;
		video::CDumbPresentationOracle m_oracle;

		uint16_t gcIndex = {}; // note: this is dirty however since I assume only single object in scene I can leave it now, when this example is upgraded to support multiple objects this needs to be changed

		bool move = false;
		float fov = 60.f, zNear = 0.1f, zFar = 10000.f, moveSpeed = 1.f, rotateSpeed = 1.f;
		float viewWidth = 10.f;
		float camYAngle = 165.f / 180.f * 3.14159f;
		float camXAngle = 32.f / 180.f * 3.14159f;

		bool m_firstFrame = true;
		IGPUCommandBuffer::SClearColorValue clearColor = { .float32 = {0.f,0.f,0.f,1.f} };
};

NBL_MAIN_FUNC(ComputeShaderPathtracer)
