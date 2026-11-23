KPL/MK

   Meta-kernel of the SPICE kernels `earth_visibility.py` loads (Earth visibility at the lunar base site).
   The kernels themselves are not tracked: `space_context/fetch_kernels.sh` downloads them from the NAIF generic
   kernels into space_context/kernels/ and checks them against space_context/kernels.md5. Load this file with
   furnsh from the space_context directory (PATH_VALUES is relative to it).

   naif0012.tls               leap seconds                          lsk/
   pck00011.tpc               planetary constants (Moon radii)      pck/
   de440s.bsp                 planetary ephemeris DE440 (short)     spk/planets/
   moon_pa_de440_200625.bpc   lunar orientation, principal axes     pck/
   moon_de440_250416.tf       lunar frames (MOON_PA -> MOON_ME)      fk/satellites/

\begindata

   PATH_VALUES     = ( 'kernels' )
   PATH_SYMBOLS    = ( 'K' )
   KERNELS_TO_LOAD = ( '$K/naif0012.tls'
                       '$K/pck00011.tpc'
                       '$K/de440s.bsp'
                       '$K/moon_pa_de440_200625.bpc'
                       '$K/moon_de440_250416.tf' )

\begintext
