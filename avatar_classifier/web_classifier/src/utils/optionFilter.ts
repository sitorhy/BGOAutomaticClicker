// 下拉过滤工厂：空格分隔多关键词取交集 AND，匹配 文件名+路径
// 供 HomeView 的头像路径 / 掩码图下拉与各功能子组件的路径下拉共用
import {computed, ref} from 'vue'
import type {AssetItem} from '../stores/assetStore'

export function makeOptionFilter(source: () => AssetItem[]) {
    const query = ref('')
    const filtered = computed(() => {
        const kws = query.value.trim().toLowerCase().split(/\s+/).filter(Boolean)
        if (kws.length === 0) return source()
        return source().filter(a => {
            const haystack = `${a.file} ${a.path}`.toLowerCase()
            return kws.every(kw => haystack.includes(kw))
        })
    })
    // 关闭下拉后重置关键词，下次打开显示全量选项
    function onDropdownVisible(visible: boolean) {
        if (!visible) query.value = ''
    }

    return {query, filtered, onDropdownVisible}
}
