<script setup lang="ts">
import {computed, h, onMounted, onUnmounted, ref} from 'vue'
import {NButton, NDataTable, NIcon, NInput, NSwitch, useMessage} from 'naive-ui'
import type {DataTableColumns} from 'naive-ui'
import {CheckCircleFilled} from '@vicons/material'
import {ApiManager} from '../api'
import {useStore} from '../store/useStore'
import type {Servant} from '../types'

const store = useStore()
const message = useMessage()
const search = ref('')
const unmarkedOnly = ref(false)
const togglingIds = ref<Set<number>>(new Set())

const listBodyRef = ref()
const listMaxHeight = ref(400)

function isFullyMarked(s: Servant): boolean {
  const stats = store.getStats(s.id)
  return !!stats && stats.imageCount > 0 && stats.markedCount >= stats.imageCount
}

const filteredServants = computed(() => {
  const q = search.value.trim().toLowerCase()
  const list = store.state.servants as unknown as Servant[]
  let result = list
  if (q) {
    result = result.filter(
        (s) =>
            s.name_cn.toLowerCase().includes(q) ||
            s.name_en.toLowerCase().includes(q) ||
            String(s.id).includes(q),
    )
  }
  if (unmarkedOnly.value) {
    result = result.filter((s) => !isFullyMarked(s))
  }
  return result
})

async function onToggleRevoke(row: Servant) {
  if (togglingIds.value.has(row.id)) return
  togglingIds.value.add(row.id)
  try {
    const updated: Servant = {...row, revoke: row.revoke === 1 ? 0 : 1}
    await ApiManager.servantsService.updateServant(row.id, updated)
    store.updateServantRevoke(row.id, updated.revoke)
    if (store.state.selectedServantId === row.id) {
      await store.refreshServantAvatars(row.id)
    }
    message.success(updated.revoke === 1 ? '已屏蔽' : '已录入')
  } catch (err: any) {
    if (err?.response?.status === 404) {
      message.error('从者不存在')
    } else {
      message.error(`操作失败：${err.message ?? err}`)
    }
  } finally {
    togglingIds.value.delete(row.id)
  }
}

const columns: DataTableColumns<Servant> = [
  {title: 'id', key: 'id', width: 50, sorter: 'default'},
  {
    title: '从者',
    key: 'name_cn',
    sorter: (a, b) => a.name_cn.localeCompare(b.name_cn, 'zh'),
    render: (row) => {
      return h('span', {style: row.revoke === 1 ? {color: '#ff4d4f'} : undefined}, row.name_cn)
    }
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
  {
    title: '操作',
    key: 'actions',
    width: 70,
    fixed: 'right',
    render: (row) => {
      const isBlocked = row.revoke === 1
      const isLoading = togglingIds.value.has(row.id)
      return h(NButton, {
        size: 'tiny',
        type: isBlocked ? 'primary' : 'error',
        loading: isLoading,
        onClick: (e: Event) => {
          e.stopPropagation()
          onToggleRevoke(row)
        },
      }, {default: () => isBlocked ? '录入' : '屏蔽'})
    },
  },
]

function onRowClick(row: Servant) {
  store.selectServant(row.id)
  store.loadImages(String(row.id))
}

function rowProps(row: Servant) {
  const isSelected = store.state.selectedServantId === row.id
  return {
    style: {
      cursor: 'pointer',
    },
    class: isSelected ? ['selected'] : [],
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
      <NSwitch v-model:value="unmarkedOnly" size="small" class="unmarked-switch">
        <template #checked>标记</template>
        <template #unchecked>全部</template>
      </NSwitch>
    </div>
    <div class="servant-list-body" ref="listBodyRef">
      <NDataTable
          class="servant-table"
          :columns="columns"
          :data="filteredServants"
          :row-props="rowProps"
          :bordered="false"
          size="small"
          :scroll-x="340"
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
  display: flex;
  align-items: center;
  gap: 8px;
}

.unmarked-switch {
  flex-shrink: 0;
}

.servant-list-body {
  flex: 1;
  overflow: hidden;
}

.servant-table {
  height: 100%;
}
</style>

<style>
.servant-table .n-data-table-tr.selected td {
  color: blue;
}
</style>
