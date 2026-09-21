import { defineStore } from 'pinia'
import { reactive } from 'vue'

// 图片序列网格配置：行数 / 列数
export interface GridConfig {
  rows: number
  cols: number
}

// 默认网格：4 行 4 列（功能区表单的默认值）
const DEFAULT_GRID: GridConfig = { rows: 4, cols: 4 }

// 图片序列功能的公共状态：
// 各功能（tabId）的行列配置由 Right 功能区表单写入，Tab 区域组件读取后透传给公共组件
export const useImageSequenceStore = defineStore('imageSequence', () => {
  const grids = reactive<Record<string, GridConfig>>({})

  // 取某功能的网格配置（可写的响应式对象），不存在时初始化默认值
  function gridOf(tabId: string): GridConfig {
    if (!grids[tabId]) {
      grids[tabId] = { ...DEFAULT_GRID }
    }
    return grids[tabId]
  }

  return { grids, gridOf }
})
