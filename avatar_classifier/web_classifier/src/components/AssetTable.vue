<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { Refresh, Search } from '@element-plus/icons-vue'

export interface AssetItem {
  file: string
  path: string
  url: string
}

const props = defineProps<{
  assets: AssetItem[]
}>()

const emit = defineEmits<{
  (e: 'select', item: AssetItem | null): void
}>()

const loading = ref(false)

function handleRefresh() {
  loading.value = true
  setTimeout(() => {
    loading.value = false
  }, 500)
}

// —— 本地过滤：关键词命中 文件名 或 路径 任一即保留，空格分隔多关键词取交集 ——
const keyword = ref('')

const filteredAssets = computed(() => {
  const kws = keyword.value.trim().toLowerCase().split(/\s+/).filter(Boolean)
  if (kws.length === 0) return props.assets
  return props.assets.filter(a => {
    const haystack = `${a.file} ${a.path}`.toLowerCase()
    return kws.every(kw => haystack.includes(kw))
  })
})

// —— 本地分页：对过滤结果切片 ——
const currentPage = ref(1)
const pageSize = ref(20)
const pageSizes = [10, 20, 50, 100]

const pagedAssets = computed(() => {
  const start = (currentPage.value - 1) * pageSize.value
  return filteredAssets.value.slice(start, start + pageSize.value)
})

// 关键词变化时回到第一页
watch(keyword, () => {
  currentPage.value = 1
})

// 数据量或页大小变化时，防止当前页越界
watch([() => filteredAssets.value.length, pageSize], () => {
  const maxPage = Math.max(1, Math.ceil(filteredAssets.value.length / pageSize.value))
  if (currentPage.value > maxPage) currentPage.value = maxPage
})

// 单选：表格当前行变化时向父组件抛出选中项
function handleCurrentChange(row: AssetItem | null) {
  emit('select', row)
}
</script>

<template>
  <div class="asset-table-component">
    <!-- 上区域：刷新按钮 + 关键词过滤 -->
    <div class="top-area">
      <el-button type="primary" :loading="loading" @click="handleRefresh">
        <el-icon>
          <Refresh />
        </el-icon>
        刷新
      </el-button>
      <el-input
        v-model="keyword"
        class="filter-input"
        placeholder="过滤（空格分隔多关键词）"
        clearable
      >
        <template #prefix>
          <el-icon>
            <Search />
          </el-icon>
        </template>
      </el-input>
    </div>

    <!-- 中区域：表格 -->
    <div class="middle-area">
      <el-table
        :data="pagedAssets"
        v-loading="loading"
        stripe
        border
        highlight-current-row
        style="width: 100%"
        @current-change="handleCurrentChange"
      >
        <!-- 文件列：加宽，移除 tooltip，长文件名换行完整显示 -->
        <el-table-column prop="file" label="文件" :min-width="240" />
        <el-table-column prop="path" label="路径" :width="300" show-overflow-tooltip />
      </el-table>
    </div>

    <!-- 下区域：本地分页 -->
    <div class="bottom-area">
      <el-pagination
        v-model:current-page="currentPage"
        v-model:page-size="pageSize"
        :page-sizes="pageSizes"
        :total="filteredAssets.length"
        layout="total, sizes, prev, pager, next"
        size="small"
        background
      />
    </div>
  </div>
</template>

<style scoped>
.asset-table-component {
  height: 100%;
  display: flex;
  flex-direction: column;
  padding: 8px;
  box-sizing: border-box;
}

.top-area {
  padding-bottom: 8px;
  border-bottom: 1px solid #e0e0e0;
  display: flex;
  flex-direction: column;
  gap: 8px;
  align-items: stretch;
}

.filter-input {
  width: 100%;
}

.middle-area {
  flex: 1;
  overflow: auto;
  padding: 8px 0;
}

.bottom-area {
  flex-shrink: 0;
  min-height: 40px;
  padding-top: 8px;
  display: flex;
  justify-content: center;
}
</style>
