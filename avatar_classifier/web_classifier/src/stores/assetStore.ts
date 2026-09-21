import { defineStore } from 'pinia'
import { ref } from 'vue'
import ResourcesApi from "../api/ResourcesApi.ts";

export interface AssetItem {
  file: string
  path: string
  url: string
}

export const useAssetStore = defineStore('asset', () => {
  const assets = ref<AssetItem[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)
  const resApi = new ResourcesApi();

  async function fetchAssets(): Promise<AssetItem[]> {
    const res= await resApi.getAssetsImages("assets");
    const subdirs = res.data.subdirs;
    const filesRes = await Promise.all(subdirs.map(async dir => {
      const dirRes = await resApi.getAssetsImages(`assets/${dir}`);
      return dirRes.data.files;
    }));

    return filesRes.flat().map(f => ({ file: f.name, path: f.path, url: f.url }));
  }

  async function fetchMaskAssets(): Promise<AssetItem[]> {
    const res= await resApi.getAssetsImages("mask");
    return res.data.files.map(f => ({ file: f.name, path: f.path, url: f.url }));;
  }

  async function fetchForegroundAssets(): Promise<AssetItem[]> {
    const res= await resApi.getAssetsImages("foreground");
    return res.data.files.map(f => ({ file: f.name, path: f.path, url: f.url }));;
  }

  function setAssets(newAssets: AssetItem[]) {
    assets.value = newAssets
  }

  function setLoading(value: boolean) {
    loading.value = value
  }

  function setError(msg: string) {
    error.value = msg
  }

  return { assets, loading, error, fetchAssets, fetchMaskAssets, fetchForegroundAssets, setAssets, setLoading, setError }
})
