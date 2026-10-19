import {reactive, readonly} from 'vue'
import type {Servant, ServantAvatar, BBoxRange, ServantStats, QueueStep} from '../types'
import {ApiManager, config} from '../api'

const IMAGE_BASE = `${config.origin}/file`
const CONCURRENCY = 3

interface StoreState {
    servants: Servant[]
    selectedServantId: number | null
    images: string[]
    selectedImage: string | null
    range: BBoxRange | null
    stats: Record<number, ServantStats>
    avatarMap: Record<string, ServantAvatar>
    queueTotal: number
    queueDone: number
    processing: Record<number, QueueStep>
    initialized: boolean
}

const state = reactive<StoreState>({
    servants: [],
    selectedServantId: null,
    images: [],
    selectedImage: null,
    range: null,
    stats: {},
    avatarMap: {},
    queueTotal: 0,
    queueDone: 0,
    processing: {},
    initialized: false,
})

const imageCache: Record<number, string[]> = {}

async function fetchImages(servantId: number): Promise<string[]> {
    if (imageCache[servantId]) return imageCache[servantId]
    const res = await ApiManager.resourceService.listDir(`assets/${servantId}`)
    const names = res.data.entries
        .filter((e) => e.type === 'file' && /\.(png|jpg|jpeg|gif|webp)$/i.test(e.name))
        .map((e) => e.name)
    imageCache[servantId] = names
    return names
}

async function fetchAvatars(servantId: number): Promise<ServantAvatar[]> {
    const res = await ApiManager.servantsService.getServantAvatars(servantId)
    return res.data
}

async function processServant(servantId: number) {
    state.processing[servantId] = 'fetching_images'
    const images = await fetchImages(servantId)

    state.processing[servantId] = 'fetching_avatars'
    const avatars = await fetchAvatars(servantId)

    const markedFiles = new Set(avatars.map((a) => a.file))
    const markedCount = images.filter((name) => markedFiles.has(name)).length

    for (const avatar of avatars) {
        state.avatarMap[`${avatar.servant_id}/${avatar.file}`] = avatar
    }

    state.stats[servantId] = {
        imageCount: images.length,
        markedCount,
        step: 'idle',
    }

    delete state.processing[servantId]
    state.queueDone++
}

async function runQueue() {
    const pending = state.servants.map((s) => s.id)
    let idx = 0

    async function next() {
        while (idx < pending.length) {
            const id = pending[idx++]
            if (state.processing[id]) continue
            await processServant(id)
        }
    }

    const workers = Array.from({length: CONCURRENCY}, () => next())
    await Promise.all(workers)
}

export function useStore() {
    async function init() {
        if (state.initialized) return
        try {
            const res = await ApiManager.servantsService.listServants()
            state.servants = res.data.sort((a, b) => a.id - b.id)
        } catch {
            state.servants = []
        }
        state.queueTotal = state.servants.length
        state.queueDone = 0
        state.initialized = true
        runQueue()
    }

    function selectServant(id: number) {
        state.selectedServantId = id
        state.selectedImage = null
        state.range = null
        state.images = []
    }

    async function loadImages(dirName: string) {
        try {
            const names = await fetchImages(Number(dirName))
            state.images = [...names].sort()
        } catch {
            state.images = []
        }
    }

    function selectImage(name: string) {
        state.selectedImage = name
        state.range = null
        if (state.selectedServantId != null) {
            const key = `${state.selectedServantId}/${name}`
            const avatar = state.avatarMap[key]
            if (avatar) {
                state.range = {
                    left: avatar.left,
                    top: avatar.top,
                    right: avatar.right,
                    bottom: avatar.bottom,
                }
            }
        }
    }

    function getImageUrl(servantId: number, imageName: string): string {
        return `${IMAGE_BASE}/assets/${servantId}/${encodeURIComponent(imageName)}`
    }

    function updateRange(range: BBoxRange | null) {
        state.range = range
    }

    function createNewRange() {
        state.range = {left: 0, top: 0, right: 100, bottom: 100}
    }

    async function saveRange() {
        if (state.selectedServantId == null || !state.selectedImage || !state.range) return
        const key = `${state.selectedServantId}/${state.selectedImage}`
        const existing = state.avatarMap[key]
        await ApiManager.servantsService.createOrUpdateServantAvatar({
            id: existing?.id,
            servant_id: state.selectedServantId,
            file: state.selectedImage,
            ...state.range,
        })
        const avatars = await fetchAvatars(state.selectedServantId)
        const markedFiles = new Set(avatars.map((a) => a.file))
        const images = imageCache[state.selectedServantId] ?? []
        const markedCount = images.filter((name) => markedFiles.has(name)).length
        for (const avatar of avatars) {
            state.avatarMap[`${avatar.servant_id}/${avatar.file}`] = avatar
        }
        state.stats[state.selectedServantId] = {
            imageCount: images.length,
            markedCount,
            step: 'idle',
        }
        selectImage(state.selectedImage)
    }

    function getStats(servantId: number): ServantStats | undefined {
        return state.stats[servantId]
    }

    function isQueueDone(): boolean {
        return state.queueDone >= state.queueTotal && state.queueTotal > 0
    }

    return {
        state: readonly(state),
        init,
        selectServant,
        loadImages,
        selectImage,
        getImageUrl,
        updateRange,
        createNewRange,
        saveRange,
        getStats,
        isQueueDone,
    }
}

export type Store = ReturnType<typeof useStore>
