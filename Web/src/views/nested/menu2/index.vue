<template>
  <div class="container-menu2">
    <div class="title">修改密码</div>
    <el-form ref="form" :model="form" label-width="100px">
      <el-form-item label="原密码" style="width:300px;margin-right:80px;">
        <el-input v-model="form.old_password" />
      </el-form-item>
      <el-form-item label="新密码" style="width:300px;">
        <el-input v-model="form.new_password" />
      </el-form-item>
      <el-form-item label="确认密码" style="width:300px;">
        <el-input v-model="form.new_confirm_password" />
      </el-form-item>
      <el-form-item>
        <el-button type="primary" @click="onSubmit">确认修改</el-button>
        <el-button>取消重置</el-button>
      </el-form-item>
    </el-form>
  </div>
</template>
<script>
export default {
  data() {
    return {

      form: {
        old_password: '',
        new_password: '',
        new_confirm_password: ''
      }

    }
  },
  created() {
    const userInfo = JSON.parse(sessionStorage.getItem('userInfo'))
    // console.log('userInfo', userInfo)
    this.form = userInfo
  },
  methods: {
    onSubmit() {
      console.log('form-userInfo', this.form)
    }
  }
}
</script>
<style >

.container-menu2{

  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  margin-top: 260px;

}

.title{

  margin-bottom: 50px;
  font-size: 26px;
  font-family: "Helvetica Neue",Helvetica,"PingFang SC","Hiragino Sans GB","Microsoft YaHei","微软雅黑",Arial,sans-serif;

}

</style>
