import numpy as np

# =============================================================================
# Supplementary functions
# =============================================================================
def straight_line_gradient(x1: float, x2: float, y1: float, y2: float) -> float:
    """
    Get gradient of a straight line, m, given two x, y co-ordinate pairs (x1, y1) & (x2, y2)

    Args:
        x1 (float): x-coordinate 1
        x2 (float): x-coordinate 2
        y1 (float): y-coordinate 1
        y2 (float): y-coordinate 2

    Returns:
        float: gradient of a straight line, m
    """
    if x2 - x1 == 0:
        return 0
    else:
        m = (y2 - y1) / (x2 - x1)
    return m


class relPermModel:
    def __init__(
        self,
        swi: float,
        swf: float,
        krwe: float,
        kroe: float,
        nw: float,
        no: float,
        muw: float,
        muo: float,
        no_rows: int = 41,
    ) -> None:
        self.swi = swi
        self.swf = swf
        self.krwe = krwe
        self.kroe = kroe  # set default = 1 ?
        self.nw = nw
        self.no = no
        self.muw = muw  # change as input to fw ?
        self.muo = muo  # change as input to fw ?
        self.no_rows = no_rows

        # Lists to be populated
        self.sw_arr = []
        self.krw_arr = []
        self.kro_arr = []
        self.fw_arr = []

        # Values to be populated when calling bl_solution() method
        self.swbt = None
        self.fwbt = None
        self.sw_avg = None

    def create_relperm_model(self) -> None:
        """Create relative permeability model"""
        self.sw_arr = self.saturation_range(self.swi, self.swf, self.no_rows)
        sw_arr_norm = self.normalise_saturation(self.sw_arr)
        self.krw_arr = self.corey_kr(sw_arr_norm, self.krwe, self.nw)
        self.kro_arr = self.corey_kr(
            [1 - swn for swn in sw_arr_norm], self.kroe, self.no
        )

    def create_fw(self):
        """Calculate fw given a relative permeability model"""
        self.fw_arr = self.calculate_fw(self.krw_arr, self.kro_arr, self.muw, self.muo)

    def bl_solution(self):
        """Find a solution to the Buckley-Leverett equation using the Welge tangent construction method"""
        m_arr = self.calculate_fw_gradient(self.fw_arr, self.sw_arr)
        self.swbt, self.fwbt, self.sw_avg = self.welge_construction(
            self.sw_arr, self.fw_arr, m_arr
        )
        self.rf = self.calculate_recovery(self.swi, self.sw_avg)

    # =============================================================================
    # Relative permeability functions
    # =============================================================================
    def saturation_range(self, swi: float, swf: float, no_rows: int = 41) -> np.array:
        """
        Create saturation array given start and end saturations.

        Args:
            swi (float): initial saturation
            swf (float): final saturation
            no_rows (int, optional): number of rows. Defaults to 41

        Returns:
            np.array: array of saturation values
        """

        sw_arr = np.linspace(swi, swf, no_rows)

        return sw_arr

    def normalise_saturation(self, sw_arr: np.array) -> list:
        """
        Normalise saturation array for use in relative permeability calculations.
        Swn = (Sw - Swi)/(1 - Swi - (1 - Swf))

        Args:
            sw_arr (np.array): array of saturation values

        Returns:
            list: array of normalised saturation values
        """

        sw_arr_norm = [
            ((sw - min(sw_arr)) / (1 - min(sw_arr) - (1 - max(sw_arr))))
            for sw in sw_arr
        ]

        return sw_arr_norm

    def corey_kr(self, sw_arr_norm: list, kre: float, n: float) -> list:
        """
        Calculate relative permeability using the Corey parameterisation.

        Args:
            sw_arr_norm (list): array of normalised saturation values
            kre (float): endpoint relative permeability
            n (float): Corey parameter

        Returns:
            list: array of relative permeability values
        """

        return [kre * sw_norm**n for sw_norm in sw_arr_norm]

    # =============================================================================
    # Fractional flow functions
    # =============================================================================
    def calculate_fw(self, krw: list, kro: list, muw: float, muo: float) -> list:
        """
        Calculate a fractional flow curve.

        Args:
            krw (list): array of relative permeability values (water)
            kro (list): array of relative permeability values (oil)
            muw (float): water viscosity
            muo (float): oil viscosity

        Returns:
            list: array of fractional flow values
        """

        fw_arr = [
            (krw[i] / muw) / (((krw[i] / muw)) + ((kro[i] / muo)))
            for i in range(0, len(krw))
        ]

        return fw_arr

    def calculate_fw_gradient(self, fw_arr: list, sw_arr: list) -> list:
        """
        Calculate the gradient for the line from Swi to each point in the fractional flow curve

        Args:
            fw_arr (list): array of fractional flow values
            sw_arr (list): array of saturation values

        Returns:
            list: array of gradients of the line from Swi to each point in the fractional flow curve
        """
        sw_swi_arr = [sw - min(sw_arr) for sw in sw_arr]
        m_arr = [(i / j if i > 0 else 0) for i, j in zip(fw_arr, sw_swi_arr)]

        return m_arr

    def calculate_swavg(self, m_max: float, swi: float) -> float:
        """
        Calculate the average saturation behind the flood front (Swavg).
        The straight-line equation y2-y1/x2-x1 = m is rearranged to get x2,
        which is Swavg

        Args:
            m_max (float): gradient of the tangent to the fractional flow curve
            swi (float): initial saturation

        Returns:
            float: average water saturation behind the flood front
        """

        x1, y1 = swi, 0
        y2 = 1

        return ((y2 - y1) / m_max) + x1

    def welge_construction(self, sw_arr: list, fw_arr: list, m_arr: list) -> tuple:
        """
        Calculate the saturation at the flood front (at breakthrough), the fractional flow at the flood front
        (water cut at breakthrough, reservoir units) and the average saturation behind the flood front

        Args:
            sw_arr (list): array of saturation values
            fw_arr (list): array of fractional flow values
            m_arr (list): array of gradients of the line from Swi to each point in the fractional flow curve

        Returns:
            dict: tuple with values for Swbt, fwbt and Swavg
        """

        # the tangent to the fw curve defines the solution, and is the maximum gradient
        m_max = max(m_arr)
        max_m_index = m_arr.index(m_max)

        swbt = sw_arr[max_m_index]
        fwbt = fw_arr[max_m_index]
        sw_avg = self.calculate_swavg(m_max, sw_arr[0])

        return swbt, fwbt, sw_avg

    def calculate_recovery(self, swi: float, sw_avg: float) -> float:
        """
        Calculate recovery factor from the initial saturation and average saturation behind the flood front

        Args:
            swi (float): initial saturation
            sw_avg (float): average saturation behind the flood front

        Returns:
            float: recovery factor represented as a fraction
        """
        return (sw_avg - swi) / (1 - swi)


if __name__ == "__main__":
    kr = relPermModel(0.1, 0.8, 0.5, 1, 2, 2, 1, 0.35)
    kr.create_relperm_model()
    kr.create_fw()
    kr.bl_solution()
    print(kr.sw_avg, kr.rf)
