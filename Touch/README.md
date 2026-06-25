# Touch tracker

Intercepts windows touch layer, catalogues individual touches, sends packet of bytes via UDP.

## Dependencies

Annoyingly requires python 3.11, and pywin32. Easily run with "py -3.11 touchlogger4_udp.py"

## Running

Runs an invisible UI layer to intercept touches. Be sure you've disabled 3 and 4 finger gestures through windows. To toggle the intercept layer on and off, press Ctrl+Shift+T.

If dealing with differing aspect ratios between IR frame and screen, you can submit the physical measurements of both, align the *top left corners* of both, and add args when running, e.g. "py -3,11 touchlogger4_udp.py --screen-phys 310 490 --frame-phys 420 510" and it will perform a remap.