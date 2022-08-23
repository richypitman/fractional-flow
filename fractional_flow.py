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
        no_rows: int = 41,
    ) -> None:
        self.swi = swi
        self.swf = swf
        self.krwe = krwe
        self.kroe = kroe  # set default = 1 ?
        self.nw = nw
        self.no = no
        self.no_rows = no_rows

        # Lists to be populated
        self.sw_arr = []
        self.krw_arr = []
        self.kro_arr = []

        assert 0 < swi < 1, "Swi must be greater than 0"
        assert 0 < 1 - swf < 1, "Swf must be greater than 0"
        assert (1 - swi - (1 - swf)) > 0, "Swi + Swf must be greater than 0"

    def create_relperm_model(self) -> None:
        """Create relative permeability model"""
        self.sw_arr = self.saturation_range(self.swi, self.swf, self.no_rows)
        sw_arr_norm = self.normalise_saturation(self.sw_arr)
        self.krw_arr = self.corey_kr(sw_arr_norm, self.krwe, self.nw)
        self.kro_arr = self.corey_kr(
            [1 - swn for swn in sw_arr_norm], self.kroe, self.no
        )

    # =============================================================================
    # Relative permeability functions
    # =============================================================================
    def saturation_range(self, swi: float, swf: float, no_rows: int = 41) -> list:
        """
        Create saturation array given start and end saturations.

        Args:
            swi (float): initial saturation
            swf (float): final saturation
            no_rows (int, optional): number of rows. Defaults to 41

        Returns:
            np.array: array of saturation values
        """

        sw_arr = np.linspace(swi, swf, no_rows).tolist()

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


class fractionalFlowCurve:
    def __init__(
        self, sw_arr: list, krw_arr: list, kro_arr: list, muw: float, muo: float
    ) -> None:
        self.sw_arr = sw_arr
        self.krw_arr = krw_arr
        self.kro_arr = kro_arr
        self.muw = muw
        self.muo = muo

        # Lists to be populated
        self.fw_arr = []
        self.m_arr = []

        # Values to be updated
        self.swi = min(sw_arr)
        self.swf = max(sw_arr)
        self.swbt = None
        self.fwbt = None
        self.sw_avg = None

    def create_fw(self):
        """Calculate fw given a relative permeability model"""
        self.fw_arr = self.calculate_fw(self.krw_arr, self.kro_arr, self.muw, self.muo)

    def perform_welge_construction(self):
        self.m_arr = self.calculate_fw_gradient(self.fw_arr, self.sw_arr)
        self.swbt, self.fwbt, self.sw_avg = self.welge_construction(
            self.sw_arr, self.fw_arr, self.m_arr
        )

    # =============================================================================
    # Fractional flow functions
    # =============================================================================
    def calculate_fw(
        self, krw_arr: list, kro_arr: list, muw: float, muo: float
    ) -> list:
        """
        Calculate a fractional flow curve.

        Args:
            krw_arr (list): array of relative permeability values (water)
            kro_arr (list): array of relative permeability values (oil)
            muw (float): water viscosity
            muo (float): oil viscosity

        Returns:
            list: array of fractional flow values
        """

        fw_arr = [
            (krw_arr[i] / muw) / (((krw_arr[i] / muw)) + ((kro_arr[i] / muo)))
            for i in range(0, len(krw_arr))
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


class fractionalFlowGradient:
    def __init__(self, fw_arr: list, sw_arr: list) -> None:
        self.fw_arr = fw_arr
        self.sw_arr = sw_arr

        # Lists to be populated
        self.dfw_dsw_arr = []

    def create_dfw_dsw(self):
        self.dfw_dsw_arr = self.calculate_dfw_dsw(self.fw_arr, self.sw_arr)

    def calculate_dfw_dsw(self, fw_arr: list, sw_arr: list) -> list:
        """
        Calculate the gradient of the fractional flow curve, dfw/dSw

        Args:
            fw_arr (list): array of fractional flow values
            sw_arr (list): array of saturation values

        Returns:
            list: array of dfw/dSw values
        """
        dfw_dsw_arr = np.gradient(fw_arr, sw_arr).tolist()

        return dfw_dsw_arr


class shockFront:
    def __init__(
        self, sw_arr: list, dfw_dsw_arr: list, swi: float, swbt: float, swf: float
    ) -> None:
        self.sw_arr = sw_arr
        self.dfw_dsw_arr = dfw_dsw_arr
        self.swi = swi
        self.swbt = swbt
        self.swf = swf

        # Lists to be populated
        self.x_arr = []
        self.sw_shock_arr = []

    def create_shock_front(self):
        self.x_arr, self.sw_shock_arr = self.shock_front(
            self.sw_arr, self.dfw_dsw_arr, self.swi, self.swbt, self.swf
        )

    def shock_front(
        self, sw_arr: list, dfw_dsw_arr: list, swi: float, swbt: float, swf: float
    ) -> tuple:
        """
        Create shock front plot from gradient of the fractional flow curve, dfw/dSw

        Args:
            sw_arr (list): array of saturation values
            dfw_dsw_arr (list): array of dfw/dSw values
            swi (float): initial saturation
            swbt (float): breakthrough saturation
            swf (float): final saturation

        Returns:
            tuple: tuple with two arrays: x_arr and sw_shock_arr
        """

        # Sw until swbt (flood front distance) -> x
        # Shock front from swbt to swi at xbt, xbt
        # Then swi for all x until x=1

        x_arr = dfw_dsw_arr.copy()
        sw_shock_arr = sw_arr.copy()

        index_bt = sw_shock_arr.index(swbt)  # Is this robust?
        sw_shock_arr = sw_shock_arr[index_bt:]
        sw_shock_arr.extend([swi] * 2)

        xbt = x_arr[index_bt]  # @ swi
        xmax = max(x_arr)  # @ swi
        x_arr = x_arr[index_bt:]
        x_arr.append(xbt)
        x_arr.append(xmax)

        x_arr.sort()
        sw_shock_arr.sort(reverse=True)

        x_arr = [((x - min(x_arr)) / (max(x_arr) - min(x_arr))) for x in x_arr]

        return (x_arr, sw_shock_arr)
