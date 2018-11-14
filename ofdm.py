from itertools import zip_longest

import numpy as np
from bitarray import bitarray

from matplotlib import pyplot as plt
from constants import QAM_POINTS, QAM16, SYMBOL_SIZE


class OfdmStages(object):

    def __init__(self, n_carriers=400, n_fft=1024, file_path='ofdm.png', constellation=None):
        """
        :param n_carriers:
        :param n_fft:
        :param file_path: путь до файла
        :param constellation:
        """
        self.n_carriers = n_carriers
        self.n_fft = n_fft
        self.file_path = file_path
        self.constellation = constellation

    # OFDM Transmitter methods

    def bit_reader(self) -> bitarray:
        """Читает битовое сообщение из файла
            :return экземпляр класса bitarray
        """
        bit_string = bitarray()
        with open(self.file_path, 'rb') as file:
            bit_string.fromfile(file)
        symbol_number = bit_string.length() // SYMBOL_SIZE
        slice_limit = symbol_number * SYMBOL_SIZE
        return bit_string[:slice_limit]

    def mapper(self, bit_array=None) -> np.array:
        """Преобразует bit_array в лист точек на комплексной плоскости
            :return: list: np.array с комплексными точками
        """
        bit_array_decoded = bit_array.decode(self.constellation)
        complex_list = list(map(complex, bit_array_decoded))

        return np.array(complex_list)

    def grouper(self, n, iterable=None, fillvalue=np.complex128(0)) -> np.array:
        """grouper(2, [1,2,3,4,5,6,7], 0) --> [[1,2],[3,4],[5,6],[7,0]]
            :param iterable: массив комплексных точек
            :param fillvalue: значение, которым мы дополним последний не влезший массив
            :return: np.array[np.array..np.array] массив массивов размера n_carriers
        """
        return np.array(list(zip_longest(*[iter(iterable)]*n, fillvalue=fillvalue)))

    def upsampler(self, groups=None) -> np.array:
        """upsampler([[1,2,3],[1,2,3]], N_fft=5) --> [[0,1,2,3,0], [0,1,2,3,0]]
            :param groups: Группы массивов комплексных точек
            :return: np.array[np.array..np.array] массив массивов размера N_fft c первым нулем
        """
        list_of_lists = list()

        for group in groups:
            new_group = np.insert(group, 0, 0)
            group_len = self.n_fft - new_group.size
            new_group = np.append(new_group, np.zeros(group_len))
            list_of_lists.append(new_group)

        return np.array(list_of_lists)

    @staticmethod
    def ifft_transmitter(upsampled=None) -> np.array:
        """Обратное преобразование Фурье для каждого массива из np.array
            :param upsampled: np.array[np.array..np.array] массив массивов размера N_fft c первым нулем
            :return np.array[np.array..np.array] пропущенное через ОДПФ
        """
        ifft = list(map(np.fft.ifft, upsampled))
        return np.array(ifft)

    @staticmethod
    def stitching(ifft_transmitted=None) -> np.array:
        """Сшивка ifft мекторов в ofdm сигнал
            :param ifft_transmitted: np.array[np.array..np.array] пропущенное через ОДПФ
            :return ofdm_signal: np.array развернутый ifft_transmitted
        """
        return np.reshape(ifft_transmitted, -1)

    # OFDM receiver methods
    def restitching(self, ofdm_signal) -> np.array:
        """Расшивка по массивам размера n_fft

            :param ofdm_signal: np.array развернутый ifft_transmitted
            :return: np.array[np.array..np.array] вектор векторов по n_fft т.е ofdm-символы
        """
        return self.grouper(n=self.n_fft, iterable=ofdm_signal, fillvalue=np.complex128(0))

    def fft_transmitter(self, ofdm_simbols=None) -> np.array:
        """ДПФ для каждого ofdm - символа

            :param ofdm_simbols: np.array[np.array..np.array] вектор векторов т.е ofdm-символы
            :return: np.array[np.array..np.array] пропущенное через преобразование Фурье
        """
        fft = list(map(np.fft.fft, ofdm_simbols))
        plt.plot(list(map(lambda x: np.square(np.abs(x)), fft[-1])), 'ro')
        plt.show()
        return np.array(fft)

    def extract_information_frequency_band(self, fft_transmitted=None) -> np.array:
        """Выделение информационной полосы частот
            :param fft_transmitted: np.array[np.array..np.array] пропущенное через преобразование Фурье
            :return: np.array[np.array..np.array] размером по n_carriers каждый
        """
        extract = list()

        for fft_list in fft_transmitted:
            extract.append(fft_list[1:self.n_carriers+1])

        return np.array(extract)

    def demapper(self, groups=None) -> bitarray: # реализовать
        """Отображение комплексных чисел в двоичный битовый вид
            :param groups: np.array[np.array..np.array]
            :param eps: фильтрующее нули значение
            :return: bitarray
        """
        message = np.reshape(groups, -1)
        dist_arr = abs(np.asarray(message).reshape((-1, 1)) - np.asarray(QAM_POINTS).reshape((1, -1)))
        min_arg = dist_arr.argmin(axis=1)
        hard_decidion = np.asarray(QAM_POINTS)[min_arg]
        bit_message = bitarray()

        ms = list(map(lambda x: '{:.1f}'.format(x), hard_decidion))
        bit_message.encode(self.constellation, ms)

        return bit_message

    @staticmethod
    def scrambler(input_array=None, register=bitarray('100101010000000')) -> bitarray:
        """Рандомизатор входной(выходной) последовательности
            :param input_array: входящий поток
            :param register: инициализирующая последовательность
            :return: рандомизированная последовательность
        """
        output_bit_array = bitarray()
        temp_input_array = input_array.copy()
        temp_register = register.copy()
        register_size = temp_register.length()

        temp_input_array.reverse()

        for bit in temp_input_array:
            last_bit_xor = temp_register[register_size - 1]
            pre_last_bit_xor = temp_register[register_size - 2]
            xored_bit = pre_last_bit_xor ^ last_bit_xor
            temp_register.insert(0, xored_bit)
            temp_register.pop()
            input_bit_xor = xored_bit ^ bit
            output_bit_array.insert(0, input_bit_xor)

        return output_bit_array


if __name__ == "__main__":
    ofdm = OfdmStages(constellation=QAM16, file_path='image.png')
    bit_arr = ofdm.bit_reader()
    print(bit_arr.length())
    bit_arr_scram = ofdm.scrambler(input_array=bit_arr)
    mapped_list = ofdm.mapper(bit_array=bit_arr_scram)
    groups = ofdm.grouper(ofdm.n_carriers, iterable=mapped_list)
    upsampled = ofdm.upsampler(groups=groups)
    ifft_transmitted = ofdm.ifft_transmitter(upsampled=upsampled)
    ofdm_signal = ofdm.stitching(ifft_transmitted=ifft_transmitted)
    ofdm_simbols = ofdm.restitching(ofdm_signal=ofdm_signal)
    fft_transmitted = ofdm.fft_transmitter(ofdm_simbols=ofdm_simbols)
    groups_final = ofdm.extract_information_frequency_band(fft_transmitted=fft_transmitted)
    bit_arr_end = ofdm.demapper(groups=groups_final)
    bit_arr_suka = ofdm.scrambler(input_array=bit_arr_end)
    print(bit_arr.length())
    print(bit_arr_suka.length())
    print(bit_arr ^ bit_arr_suka)




