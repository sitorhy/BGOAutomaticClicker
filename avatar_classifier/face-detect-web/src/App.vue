<script setup lang="ts">
import {onMounted} from 'vue'
import {useStore} from './store/useStore'
import Header from './components/Header.vue'
import ServantList from './components/ServantList.vue'
import ImageCanvas from './components/ImageCanvas.vue'
import ImagePaths from './components/ImagePaths.vue'
import RangeDisplay from './components/RangeDisplay.vue'
import StatusBar from './components/StatusBar.vue'

const store = useStore()

onMounted(() => {
  store.init()
})
</script>

<template>
  <n-message-provider>
    <div class="app">
      <Header/>
      <div class="app-body">
        <div class="panel-left">
          <ServantList/>
        </div>
        <div class="panel-center">
          <ImageCanvas/>
        </div>
        <div class="panel-right">
          <ImagePaths/>
        </div>
      </div>
      <div class="app-footer">
        <RangeDisplay/>
        <StatusBar/>
      </div>
    </div>
  </n-message-provider>
</template>

<style scoped>
.app {
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
}

.app-body {
  display: flex;
  flex: 1;
  overflow: hidden;
}

.panel-left {
  width: 450px;
  min-width: 220px;
  overflow: hidden;
}

.panel-center {
  flex: 1;
  overflow: hidden;
  display: flex;
}

.panel-right {
  width: 400px;
  min-width: 200px;
  overflow: hidden;
}

.app-footer {
  flex-shrink: 0;
}
</style>
