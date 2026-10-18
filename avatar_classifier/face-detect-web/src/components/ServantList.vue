<script setup lang="ts">
import {computed, h, onMounted, onUnmounted, ref} from 'vue'
import {NDataTable, NIcon, NInput} from 'naive-ui'
import type {DataTableColumns} from 'naive-ui'
import {CheckCircleFilled} from '@vicons/material'
import {useStore} from '../store/useStore'
import type {Servant} from '../types'

const store = useStore()
const search = ref('')

const listBodyRef = ref()
const listMaxHeight = ref(400)

const filteredServants = computed(() => {
  const q = search.value.trim().toLowerCase()
  const list = store.state.servants as unknown as Servant[]
  if (!q) return list
  return list.filter(
      (s) =>
          s.name_cn.toLowerCase().includes(q) ||
          s.name_en.toLowerCase().includes(q) ||
          String(s.id).includes(q),
  )
})

const columns: DataTableColumns<Servant> = [
  {title: 'id', key: 'id', width: 50, sorter: 'default'},
  {
    title: '从者',
    key: 'name_cn',
    sorter: (a, b) => a.name_cn.localeCompare(b.name_cn, 'zh'),
  },
  {
    title: '立绘数',
    key: 'imageCount',
    width: 70,
    render: (row) => {
      const stats = store.getStats(row.id)
      return stats ? stats.imageCount : '-'
    },
  },
  {
    title: '已标记',
    key: 'markedCount',
    width: 70,
    render: (row) => {
      const stats = store.getStats(row.id)
      if (!stats) return '-'
      if (stats.imageCount > 0 && stats.markedCount >= stats.imageCount) {
        return h(NIcon, {size: 16, color: '#52c41a'}, {default: () => h(CheckCircleFilled)})
      }
      return `${stats.markedCount}`
    },
  },
]

function onRowClick(row: Servant) {
  store.selectServant(row.id)
  store.loadImages(String(row.id))
}

function rowProps(row: Servant) {
  return {
    style: {
      cursor: 'pointer',
      background:
          store.state.selectedServantId === row.id ? '#e6f4ff' : undefined,
    },
    onClick: () => onRowClick(row),
  }
}

onMounted(() => {
  listMaxHeight.value = listBodyRef.value.clientHeight - 40
  window.addEventListener('resize', onWindowResize)
})

onUnmounted(() => {
  window.removeEventListener('resize', onWindowResize)
})

const onWindowResize = () => {
  listMaxHeight.value = listBodyRef.value.clientHeight - 40
}
</script>

<template>
  <div class="servant-list">
    <div class="servant-list-header">
      <NInput v-model:value="search" placeholder="搜索从者..." size="small" clearable/>
    </div>
    <div class="servant-list-body" ref="listBodyRef">
      <NDataTable
          :columns="columns"
          :data="filteredServants"
          :row-props="rowProps"
          :bordered="false"
          size="small"
          :scroll-x="200"
          :max-height="listMaxHeight"
      />
    </div>
  </div>
</template>

<style scoped>
.servant-list {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: #fff;
  border-right: 1px solid #e0e0e0;
}

.servant-list-header {
  padding: 8px;
  border-bottom: 1px solid #f0f0f0;
}

.servant-list-body {
  flex: 1;
  overflow: hidden;
}

.servant-table {
  height: 100%;
}
</style>
