__author__ = 'aymgal'


from MuSLIT.light.light_component import LightComponent


class SourceLight(LightComponent):
    """Class that describes a the source galaxy light distribution.
    """

    def __init__(self, num_pix,  num_bands, lens_mass_obj, random_init=True):
        super().__init__(num_pix, num_bands, random_init=random_init)
        self._lens = lens_mass_obj


    def lensed(self):
        return self._lens.source_to_image(self.light) 
