<script setup lang="ts">
// 公共组件：以 n x m 网格布局分页展示图片序列
// n = 行数（默认 9），m = 列数（默认 4），每页最多展示 n * m 张图片
// 图片数目可能大于 n * m，通过本地分页翻页查看
import { computed, ref, watch } from 'vue'

const props = withDefaults(
  defineProps<{
    // 图片序列（路径数组）
    images: string[]
    // 行数 n
    n?: number
    // 列数 m
    m?: number
  }>(),
  {
    n: 9,
    m: 4,
  }
)

// 每页容量 = n * m
const pageSize = computed(() => Math.max(1, props.n * props.m))
const pageCount = computed(() => Math.ceil(props.images.length / pageSize.value))

// 当前页（本地状态，从 1 开始）
const currentPage = ref(1)

// 页容量或数据变化时，收敛当前页，避免越界空页
watch([pageSize, () => props.images.length], () => {
  if (currentPage.value > pageCount.value) {
    currentPage.value = Math.max(1, pageCount.value)
  }
})

// 当前页要渲染的图片
const visibleImages = computed(() => {
  const start = (currentPage.value - 1) * pageSize.value
  return props.images.slice(start, start + pageSize.value)
})
</script>

<template>
  <div class="image-grid-component">
    <!-- 上部分：限定高度的预留区域 -->
    <div class="top-area">
    </div>

    <!-- 中部分：网格，组件内唯一允许滚动的区域 -->
    <div class="middle-area">
      <div
        v-if="images.length"
        class="image-grid"
        :style="{ gridTemplateColumns: `repeat(${m}, minmax(0, 1fr))` }"
      >
        <div v-for="(src, index) in visibleImages" :key="index" class="image-cell">
          <el-image :src="src" fit="contain" class="cell-image" lazy />
        </div>
      </div>
      <el-empty v-else description="暂无图片序列" />
    </div>

    <!-- 下部分：本地分页控件，固定占位、不被中部滚动挤占 -->
    <div class="bottom-area">
      <div v-if="pageCount > 1" class="image-pagination">
        <el-pagination
          v-model:current-page="currentPage"
          :page-size="pageSize"
          :total="images.length"
          layout="prev, pager, next, total"
          background
          small
        />
      </div>
    </div>
  </div>
</template>

<style scoped>
/* 弹性布局：垂直上 / 中 / 下结构，整体不滚动 */
.image-grid-component {
  height: 100%;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  padding: 8px;
  box-sizing: border-box;
}

/* 上部分：限高预留 */
.top-area {
  flex-shrink: 0;
  min-height: 40px;
}

/* 中部分：仅此区域滚动 */
.middle-area {
  flex: 1;
  min-height: 0;
  overflow: auto;
}

/* 下部分：固定高度容器，保证分页控件始终可见 */
.bottom-area {
  flex-shrink: 0;
  min-height: 40px;
  padding-top: 8px;
}

.image-grid {
  display: grid;
  gap: 8px;
}

.image-cell {
  aspect-ratio: 3 / 4;
  border: 1px solid #e0e0e0;
  border-radius: 4px;
  background-color: #fafafa;
  overflow: hidden;
}

.cell-image {
  width: 100%;
  height: 100%;
}

.image-pagination {
  display: flex;
  justify-content: center;
}
</style>
