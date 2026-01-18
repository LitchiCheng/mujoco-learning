import numpy as np

class LowPassOnlineFilter:
    def __init__(self, dimension=6, tau=0.1, dt=0.01, initial_states=None):
        # tau: 时间常数
        # dt: 采样时间（单个采样时刻的间隔）
        self.alpha = dt / (tau + dt)
        if initial_states is None:
            self.last_outputs = np.zeros(dimension)
        else:
            if initial_states.shape != (dimension,):
                raise ValueError("Initial states shape mismatch!")
            self.last_outputs = initial_states.copy()

    def update(self, input_array):
        if not isinstance(input_array, np.ndarray) or input_array.shape != self.last_outputs.shape:
            raise ValueError("Input array shape mismatch!")
        
        output_array = np.zeros(self.last_outputs.shape[0])
        for dim in range(self.last_outputs.shape[0]):
            output_array[dim] = self.alpha * input_array[dim] + (1 - self.alpha) * self.last_outputs[dim]
            self.last_outputs[dim] = output_array[dim]
        
        return output_array

if __name__ == "__main__":
    filter_6d = LowPassOnlineFilter(dimension=6)
    input_zero = np.zeros(6)
    output_zero = filter_6d.update(input_zero)
    print("6维全零输入：", input_zero)
    print("6维全零输出：", output_zero)

    filter_8d = LowPassOnlineFilter(dimension=8)
    input_8d = np.array([1,2,3,4,5,6,7,8])
    output_8d = filter_8d.update(input_8d)
    print("\n8维输入：", input_8d)
    print("8维输出：", output_8d)

    output_8d_2 = filter_8d.update(input_8d)
    print("\n再次输入8维数组输出：", output_8d_2)