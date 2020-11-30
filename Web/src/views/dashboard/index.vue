<template>
  <div class="dashboard-container">
    <WelcomPanel />
    <PanelGroup />
    <el-row style="background:#fff;padding:16px 16px 0;margin-bottom:32px;">
      <line-chart :chart-data="lineChartData" />
    </el-row>
  </div>
</template>

<script>
import { mapGetters } from 'vuex'
import PanelGroup from './components/panelGroup'
import LineChart from './components/lineChart'
import WelcomPanel from './components/welcomPanel'

export default {
  name: 'Dashboard',
  components: {
    PanelGroup,
    LineChart,
    WelcomPanel
  },
  data() {
    return {

      lineChartData: {
        hatCount: [100, 120, 161, 134, 105, 160, 165, 200, 220, 161, 184, 105, 160, 165, 261, 234, 205, 260],
        personCount: [120, 82, 91, 154, 162, 140, 145, 100, 86, 61, 134, 99, 110, 160, 161, 134, 105, 160, 165]
      }

    }
  },
  computed: {
    ...mapGetters([
      'name'
    ])
  }
}
</script>

<style lang="scss" scoped>
.dashboard {
  &-container {
    margin: 30px;
  }
  &-text {
    font-size: 30px;
    line-height: 46px;
  }
}
</style>
