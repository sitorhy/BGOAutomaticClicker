<script setup lang="ts">
import {computed, onMounted, reactive, ref} from 'vue'
import AssetTable, {type AssetItem} from '../components/AssetTable.vue'
import TabsPanel, {type TabConfig} from '../components/TabsPanel.vue'
import RightPanel from '../components/RightPanel.vue'
import RectLocateTab, {type Rect} from '../components/RectLocateTab.vue'
import RectLocatePanel from '../components/RectLocatePanel.vue'
import ImageSequenceTab from '../components/ImageSequenceTab.vue'
import ImageSequencePanel from '../components/ImageSequencePanel.vue'
import ForegroundComposePanel from '../components/ForegroundComposePanel.vue'
import {useAssetStore} from '../stores/assetStore'
import {makeOptionFilter} from '../utils/optionFilter'
import {ElMessage} from "element-plus";
import ResourcesApi from "../api/ResourcesApi.ts";

const assetStore = useAssetStore()

// 数据源来自远程接口：onMounted 时加载一次
async function loadAssets() {
  assetStore.setLoading(true)
  try {
    assetStore.setAssets(await assetStore.fetchAssets())
  } catch (e) {
    assetStore.setError(e instanceof Error ? e.message : String(e))
  } finally {
    assetStore.setLoading(false)
  }
}

onMounted(loadAssets)

// —— 掩码图数据源：来自 assetStore.fetchMaskAssets，单独加载 ——
const maskAssets = ref<AssetItem[]>([])

async function loadMaskAssets() {
  try {
    maskAssets.value = await assetStore.fetchMaskAssets()
  } catch (e) {
    console.error('Failed to fetch mask assets:', e)
  }
}

onMounted(loadMaskAssets)

// Tab 配置：label 为标签文本，slot 既是插槽名也是 v-for 的 key
const tabConfig: TabConfig[] = [
  {label: '头像定位', slot: 'tab_4'},
  {label: '状态裁剪', slot: 'tab_status_crop'},
  {label: '助战合成', slot: 'tab_support_compose'},
  {label: '头像合成', slot: 'tab_avatar_compose'},
  {label: '状态合成', slot: 'tab_status_compose'},
]

// Left 区域单选中的图片
const selectedAsset = ref<AssetItem | null>(null)

// 当前激活的 Tab（初始为第一个），用于告诉 Right 渲染哪个插槽
const activeSlot = ref(tabConfig[0]?.slot ?? '')

// —— 图片序列（助战合成 / 头像合成 / 状态合成）——
// 按目录生成 24 帧的图片序列（mock）
function mockSequence(dir: string): string[] {
  return Array.from(
      {length: 24},
      (_, i) => `/assets/sequences/${dir}/frame_${String(i + 1).padStart(2, '0')}.svg`
  )
}

const supportSequence = mockSequence('support')
const avatarSequence = mockSequence('avatar')
const statusSequence = mockSequence('status')

// —— 矩形定位类功能（头像定位 / 状态裁剪）——
// 复用同一对公共组件 RectLocateTab + RectLocatePanel，仅功能 id 与数据不同
interface RectFeature {
  slot: string
  // 头像图片路径（仅头像定位使用，选项值来自 HomeView 的头像路径下拉）
  imagePath?: string
  // 掩码图路径（仅头像定位使用）
  maskPath?: string
  maxRect: Rect
  minRect: Rect
}

const avatarCropLocatePanelFeat = reactive<RectFeature>({
  slot: 'tab_4',
  imagePath: '',
  maskPath: '',
  maxRect: {left: 40, top: 60, right: 560, bottom: 580},
  minRect: {left: 180, top: 200, right: 420, bottom: 440},
})

const statusCropLocatePanelFeat = reactive<RectFeature>({
  slot: 'tab_status_crop',
  maxRect: {left: 8, top: 12, right: 82, bottom: 96},
  minRect: {left: 24, top: 32, right: 64, bottom: 72},
})

// —— 下拉过滤：头像路径 / 掩码图共用逻辑（工厂已抽到 utils/optionFilter）——
// 头像路径下拉：从 RectLocatePanel 抽离到此处，与掩码图下拉放在一起
const avatarPathFilter = makeOptionFilter(() => assetStore.assets)
const maskPathFilter = makeOptionFilter(() => maskAssets.value)

function handleAssetSelect(item: AssetItem | null) {
  console.log('Selected asset:', item)
  selectedAsset.value = item
}

// —— 目标图片锁定（头像定位 / 状态裁剪共用）——
// Tab 的输入图由锁定变量单独控制：未锁定时跟随左侧表格选中项预览，
// 锁定后把当时的 path/url 固定下来，之后换选中项也不再影响 Tab
interface LockedTarget {
  path: string
  url: string
}

function makeTargetLock() {
  const locked = ref<LockedTarget | null>(null)

  // Tab 实际渲染的图片：锁定项优先，否则跟随当前选中
  const image = computed(() => locked.value ?? selectedAsset.value)

  function toggle() {
    if (locked.value) {
      locked.value = null
      ElMessage.info('已解锁目标图片')
      return
    }
    if (!selectedAsset.value) {
      ElMessage.warning('请先选择目标图片')
      return
    }
    const {path, url} = selectedAsset.value
    locked.value = {path, url}
    ElMessage.success(`已锁定目标图片：${selectedAsset.value.file}`)
  }

  return {locked, image, toggle}
}

const {locked: avatarLocked, image: avatarTargetImage, toggle: toggleAvatarLock} = makeTargetLock()
const {locked: statusLocked, image: statusTargetImage, toggle: toggleStatusLock} = makeTargetLock()

function handleTabChange(slot: string) {
  activeSlot.value = slot
}

// —— 头像定位检测 ——
// detectAvatarRect 返回的单个匹配结果（rect 为 [x1, y1, x2, y2]）
interface DetectMatch {
  score: number
  rect: [number, number, number, number]
}

// 请求期间按钮 Loading，并防止重复提交
const avatarLocating = ref(false)

async function locateAvatarRect() {
  if (avatarLocating.value) return
  if (!avatarCropLocatePanelFeat.imagePath) {
    ElMessage.warning('请选择头像图片')
    return
  }
  if (!avatarCropLocatePanelFeat.maskPath) {
    ElMessage.warning('选择掩码图')
    return
  }
  // 目标图片取自锁定数据，不再直接跟随表格选中项
  if (!avatarLocked.value) {
    ElMessage.warning('请先锁定目标图片')
    return
  }
  const maskPicPath = avatarCropLocatePanelFeat.maskPath
  const templatePicPath = avatarCropLocatePanelFeat.imagePath
  const targetPicPath = avatarLocked.value.path

  avatarLocating.value = true
  try {
    const res = await new ResourcesApi().detectAvatarRect(maskPicPath, templatePicPath, targetPicPath)
    const matchResultList: DetectMatch[] = res.data
    console.log(matchResultList)

    if (!Array.isArray(matchResultList) || matchResultList.length === 0) {
      ElMessage.warning('未检测到匹配结果')
      return
    }

    // 取分数最高的矩形作为最小范围
    const best = matchResultList.reduce((a, b) => (b.score > a.score ? b : a))
    const [x1, y1, x2, y2] = best.rect
    avatarCropLocatePanelFeat.minRect = {left: x1, top: y1, right: x2, bottom: y2}

    // 最大范围 = 最小范围按中心放大 2 倍（宽高等同翻倍），左/上钳到 0
    const cx = (x1 + x2) / 2
    const cy = (y1 + y2) / 2
    const halfW = x2 - x1
    const halfH = y2 - y1
    avatarCropLocatePanelFeat.maxRect = {
      left: Math.max(0, Math.round(cx - halfW)),
      top: Math.max(0, Math.round(cy - halfH)),
      right: Math.round(cx + halfW),
      bottom: Math.round(cy + halfH),
    }

    ElMessage.success(`定位成功，最高分 ${best.score.toFixed(4)}`)
  } catch (e) {
    ElMessage.error(`头像定位失败：${e instanceof Error ? e.message : String(e)}`)
  } finally {
    avatarLocating.value = false
  }
}
</script>

<template>
  <el-container class="main-frame">
    <!-- Header 区域 -->
    <el-header class="header">
      <div class="header-content">
        <h1 class="title">Web Classifier</h1>
      </div>
    </el-header>

    <!-- Content 区域 -->
    <el-container class="content">
      <!-- Left 侧边栏 -->
      <el-aside class="aside-left">
        <AssetTable :assets="assetStore.assets" @select="handleAssetSelect"/>
      </el-aside>

      <!-- Center 主内容 -->
      <el-main class="main-center">
        <TabsPanel :tabs="tabConfig" @change="handleTabChange">
          <!-- 头像定位配置 -->
          <template #tab_4>
            <RectLocateTab
                :image-path="avatarTargetImage?.url"
                v-model:max-rect="avatarCropLocatePanelFeat.maxRect"
                v-model:min-rect="avatarCropLocatePanelFeat.minRect"
            />
          </template>

          <!-- 状态裁剪配置 -->
          <template #tab_status_crop>
            <RectLocateTab
                :image-path="statusTargetImage?.url"
                v-model:max-rect="statusCropLocatePanelFeat.maxRect"
                v-model:min-rect="statusCropLocatePanelFeat.minRect"
            />
          </template>

          <!-- 图片序列：三个功能共用同一对组件，仅 tab_id 与序列数据不同 -->
          <template #tab_support_compose>
            <ImageSequenceTab tab-id="tab_support_compose" :images="supportSequence"/>
          </template>
          <template #tab_avatar_compose>
            <ImageSequenceTab tab-id="tab_avatar_compose" :images="avatarSequence"/>
          </template>
          <template #tab_status_compose>
            <ImageSequenceTab tab-id="tab_status_compose" :images="statusSequence"/>
          </template>
        </TabsPanel>
      </el-main>

      <!-- Right 侧边栏 -->
      <el-aside class="aside-right">
        <RightPanel :image-url="selectedAsset?.url" :slot-name="activeSlot">
          <!-- 头像定位功能区 -->
          <template #tab_4>
            <div class="action-block">
              <RectLocatePanel
                  v-model:max-rect="avatarCropLocatePanelFeat.maxRect"
                  v-model:min-rect="avatarCropLocatePanelFeat.minRect"
              >
                <template #form>
                  <!-- 头像路径下拉：从 RectLocatePanel 抽离，与掩码图下拉放在一起 -->
                  <el-form-item label="头像路径">
                    <el-select
                        v-model="avatarCropLocatePanelFeat.imagePath"
                        placeholder="选择图片"
                        clearable
                        filterable
                        :filter-method="(q: string) => (avatarPathFilter.query.value = q)"
                        @visible-change="avatarPathFilter.onDropdownVisible"
                        style="width: 100%"
                    >
                      <el-option
                          v-for="item in avatarPathFilter.filtered.value"
                          :key="item.path"
                          :label="item.file"
                          :value="item.path"
                      >
                        <div class="path-option">
                          <img :src="item.url" class="path-option-thumb" alt="" loading="lazy"/>
                          <span class="path-option-name">{{ item.file }}</span>
                        </div>
                      </el-option>
                    </el-select>
                  </el-form-item>
                  <el-form-item label="掩码图">
                    <el-select
                        v-model="avatarCropLocatePanelFeat.maskPath"
                        placeholder="选择图片"
                        clearable
                        filterable
                        :filter-method="(q: string) => (maskPathFilter.query.value = q)"
                        @visible-change="maskPathFilter.onDropdownVisible"
                        style="width: 100%"
                    >
                      <el-option
                          v-for="item in maskPathFilter.filtered.value"
                          :key="item.path"
                          :label="item.file"
                          :value="item.path"
                      >
                        <div class="path-option">
                          <img :src="item.url" class="path-option-thumb" alt="" loading="lazy"/>
                          <span class="path-option-name">{{ item.file }}</span>
                        </div>
                      </el-option>
                    </el-select>
                  </el-form-item>
                </template>
                <el-row :gutter="8">
                  <el-col :span="12">
                    <el-button type="primary" :loading="avatarLocating" @click="locateAvatarRect">定位头像位置</el-button>
                  </el-col>
                  <el-col :span="12">
                    <!-- 锁定后 Tab 输入图与检测目标固定为当前预览图，可再次点击解锁 -->
                    <el-button :type="avatarLocked ? 'success' : 'default'" @click="toggleAvatarLock">
                      {{ avatarLocked ? '已锁定，点击解锁' : '锁定目标图片' }}
                    </el-button>
                  </el-col>
                </el-row>
                <el-divider/>
                <!-- 前景合成子功能：与状态裁剪同一组件，传入头像定位自己的裁剪范围与锁定图 -->
                <ForegroundComposePanel
                    :mask-options="maskAssets"
                    :max-rect="avatarCropLocatePanelFeat.maxRect"
                    :min-rect="avatarCropLocatePanelFeat.minRect"
                    :target="avatarLocked"
                />
                <el-divider/>
              </RectLocatePanel>
            </div>
          </template>

          <!-- 状态裁剪功能区 -->
          <template #tab_status_crop>
            <div class="action-block">
              <RectLocatePanel
                  v-model:max-rect="statusCropLocatePanelFeat.maxRect"
                  v-model:min-rect="statusCropLocatePanelFeat.minRect"
              >
                <!-- 目标图片锁定：与头像定位同一套交互，固定状态裁剪 Tab 的输入图 -->
                <el-row>
                  <el-col :span="24">
                    <el-button :type="statusLocked ? 'success' : 'default'" @click="toggleStatusLock">
                      {{ statusLocked ? '已锁定，点击解锁' : '锁定目标图片' }}
                    </el-button>
                  </el-col>
                </el-row>
                <el-divider/>
                <!-- 前景合成子功能：独立组件，掩码数据源与头像定位共用，
                     锁定图作预览底图、裁剪范围随步进矩形插值并打印 -->
                <ForegroundComposePanel
                    :mask-options="maskAssets"
                    :max-rect="statusCropLocatePanelFeat.maxRect"
                    :min-rect="statusCropLocatePanelFeat.minRect"
                    :target="statusLocked"
                />
                <el-divider/>
              </RectLocatePanel>
            </div>
          </template>

          <!-- 图片序列功能区：行 / 列选择写入 pinia，驱动对应 Tab 的网格 -->
          <template #tab_support_compose>
            <div class="action-block">
              <ImageSequencePanel tab-id="tab_support_compose"/>
            </div>
          </template>
          <template #tab_avatar_compose>
            <div class="action-block">
              <ImageSequencePanel tab-id="tab_avatar_compose"/>
            </div>
          </template>
          <template #tab_status_compose>
            <div class="action-block">
              <ImageSequencePanel tab-id="tab_status_compose"/>
            </div>
          </template>
        </RightPanel>
      </el-aside>
    </el-container>

    <!-- Footer 区域 -->
    <el-footer class="footer">
      <div class="footer-content">
        <p>Footer</p>
      </div>
    </el-footer>
  </el-container>
</template>

<style scoped>
.main-frame {
  width: 100vw;
  height: 100vh;
}

/* Header */
.header {
  background-color: #409eff;
  color: #fff;
  display: flex;
  align-items: center;
  padding: 0 20px;
}

.header-content {
  width: 100%;
  display: flex;
  align-items: center;
}

.title {
  margin: 0;
  font-size: 20px;
  font-weight: bold;
  color: #fff;
}

/* Content 容器 */
.content {
  flex: 1;
  overflow: hidden;
}

/* Left 侧边栏 */
.aside-left {
  width: 280px;
  background-color: #f5f5f5;
  border-right: 1px solid #e0e0e0;
  overflow: hidden;
}

/* Right 侧边栏 */
.aside-right {
  width: 420px;
  background-color: #f5f5f5;
  border-left: 1px solid #e0e0e0;
  overflow: hidden;
}

/* Center 主内容 */
.main-center {
  background-color: #fff;
  overflow-y: auto;
}

.action-block {
  padding: 8px 12px;
}

/* 路径/掩码图下拉选项：缩略图 + 文件名（头像定位表单两个下拉共用） */
.path-option {
  display: flex;
  align-items: center;
  gap: 8px;
  line-height: 1.2;
}

.path-option-thumb {
  width: 24px;
  height: 24px;
  flex-shrink: 0;
  object-fit: contain;
  border-radius: 2px;
  background-color: #f5f7fa;
}

.path-option-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* Footer */
.footer {
  background-color: #f5f5f5;
  color: #666;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0 20px;
}

.footer-content {
  width: 100%;
  text-align: center;
}

.footer p {
  margin: 0;
}
</style>
