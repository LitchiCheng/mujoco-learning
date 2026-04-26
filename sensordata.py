import src.mujoco_viewer as mujoco_viewer
import src.matplot as mp

class PandaEnv(mujoco_viewer.CustomViewer):
    def __init__(self, path):
        super().__init__(path, 3, azimuth=-45, elevation=-30)
        self.path = path

    def getSensorDataByName(self, sensor_name):
        sensor_id = self.model.sensor(sensor_name).id
        sensor_type = self.model.sensor(sensor_name).type
        sensor_dim = self.model.sensor(sensor_name).dim[0]
        # 获取数据在数组中的起始位置
        adr = self.model.sensor_adr[sensor_id]
        # 读取相应维度的数据
        sensor_values = self.data.sensordata[adr:adr+sensor_dim]
        return sensor_values, sensor_type, sensor_dim
    
    def runBefore(self):
        self.initial_pos = self.model.key_qpos[0].copy()
        for i in range(self.model.nq):
            self.data.qpos[i] = self.initial_pos[i]
        self.plot_manager = mp.MultiChartRealTimePlotManager()
        self.plot_manager.addNewFigurePlotter("j1_sensor_vel", "j1_sensor_vel", row=0, col=0)
        self.plot_manager.addNewFigurePlotter("j1_qvel", "j1_qvel", row=1, col=0)

    def runFunc(self):
        current_joint1_vel,_,_ = self.getSensorDataByName("joint1_vel")
        self.plot_manager.updateDataToPlotter("j1_sensor_vel", "j1_sensor_vel", current_joint1_vel[0])
        self.plot_manager.updateDataToPlotter("j1_qvel", "j1_qvel", self.data.qvel[0])

if __name__ == "__main__":
    env = PandaEnv("./model/franka_emika_panda/scene_pos.xml")
    env.run_loop()
