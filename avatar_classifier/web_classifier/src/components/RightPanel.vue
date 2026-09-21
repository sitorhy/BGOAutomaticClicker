<script setup lang="ts">
defineProps<{
  // 预览图片地址（来自 Left 区域选中的图片，单选）
  imageUrl?: string | null
  // 当前需要渲染的插槽名（来自 Center 区域激活的 Tab）
  slotName?: string
}>()
</script>

<template>
  <div class="right-panel-component">
    <!-- 顶部：图片预览 -->
    <div class="preview-area">
      <el-image
        v-if="imageUrl"
        :src="imageUrl"
        fit="contain"
        class="preview-image"
        :preview-src-list="[imageUrl]"
        preview-teleported
      />
      <el-empty v-else description="未选中图片" :image-size="48" />
    </div>

    <!-- 中部：动态插槽区域（按当前 Tab 的 slot 名渲染父组件提供的同名插槽） -->
    <div class="action-area">
      <slot v-if="slotName && $slots[slotName]" :name="slotName" />
    </div>

    <!-- 底部：闲置 -->
    <div class="bottom-area">
    </div>
  </div>
</template>

<style scoped>
.right-panel-component {
  height: 100%;
  display: flex;
  flex-direction: column;
  padding: 8px;
  box-sizing: border-box;
}

.preview-area {
  flex-shrink: 0;
  height: 220px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px solid #e0e0e0;
  border-radius: 4px;
  background-color: #fff;
  overflow: hidden;
}

.preview-image {
  width: 100%;
  height: 100%;
}

.action-area {
  flex: 1;
  overflow: auto;
  padding: 8px 0;
}

.bottom-area {
  flex-shrink: 0;
  min-height: 40px;
  padding-top: 8px;
}
</style>
