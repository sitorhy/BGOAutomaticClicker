<script setup lang="ts">
import {computed, h, ref, watch, type VNode} from 'vue'
import {NButton, NForm, NFormItem, NModal, NSelect, NSpace, NSpin} from 'naive-ui'
import type {SelectOption} from 'naive-ui'
import {ApiManager, config} from '../api'

const props = defineProps<{
  show: boolean
  servantId: number
  illustrationName: string
}>()

const emit = defineEmits<{
  (e: 'update:show', value: boolean): void
  (e: 'detect', result: {illustration: string; avatar: string | null; range: any} | null): void
}>()

const IMAGE_BASE = `${config.origin}/file`

const loading = ref(false)
const detecting = ref(false)
const avatarOptions = ref<SelectOption[]>([])
const selectedAvatar = ref<string | null>(null)

const visible = computed({
  get: () => props.show,
  set: (val) => emit('update:show', val),
})

watch(
    () => props.show,
    async (val) => {
      if (val && props.servantId) {
        await loadAvatarOptions()
      } else {
        selectedAvatar.value = null
        avatarOptions.value = []
      }
    },
)

async function loadAvatarOptions() {
  loading.value = true
  try {
    const res = await ApiManager.resourceService.listDir(`assets/${props.servantId}`, 'avatar')
    avatarOptions.value = res.data.entries.map((entry) => ({
      label: entry.name,
      value: entry.name,
    }))
  } catch {
    avatarOptions.value = []
  } finally {
    loading.value = false
  }
}

function renderOption(info: { node: VNode; option: SelectOption }): VNode {
  const avatarName = info.option.value as string
  const imgUrl = `${IMAGE_BASE}/assets/${props.servantId}/${encodeURIComponent(avatarName)}`
  return h('div', {style: {display: 'flex', alignItems: 'center', gap: '8px'}}, [
    h('img', {
      src: imgUrl,
      style: {width: '32px', height: '32px', objectFit: 'cover', borderRadius: '4px', flexShrink: '0'},
    }),
    h('div', {style: {flex: 1, overflow: 'hidden'}}, [info.node]),
  ])
}

async function handleConfirm() {
  visible.value = false
  detecting.value = true
  try {
    const res = await ApiManager.servantsService.detectAvatar(
        props.servantId,
        props.illustrationName,
        selectedAvatar.value,
    )
    emit('detect', {
      illustration: props.illustrationName,
      avatar: selectedAvatar.value,
      range: res.data,
    })
  } catch {
    emit('detect', null)
  } finally {
    detecting.value = false
  }
}

function handleCancel() {
  visible.value = false
}
</script>

<template>
  <NModal
      v-model:show="visible"
      preset="card"
      title="头像探测"
      :mask-closable="false"
      :closable="true"
      style="width: 460px"
      @close="handleCancel"
  >
    <NSpin :show="loading || detecting">
      <NForm label-placement="left" label-width="80">
        <NFormItem label="立绘">
          <span>{{ illustrationName }}</span>
        </NFormItem>
        <NFormItem label="头像">
          <NSelect
              v-model:value="selectedAvatar"
              :options="avatarOptions"
              :render-option="renderOption"
              placeholder="选择头像（可选）"
              clearable
              filterable
              style="width: 100%"
          />
        </NFormItem>
      </NForm>
    </NSpin>
    <template #footer>
      <NSpace justify="end">
        <NButton @click="handleCancel">取消</NButton>
        <NButton type="primary" :loading="detecting" @click="handleConfirm">确定</NButton>
      </NSpace>
    </template>
  </NModal>
</template>
