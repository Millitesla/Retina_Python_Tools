# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from psychopy import visual, core, event #import some libraries from PsychoPy

#create a window
mywin = visual.Window([2600,1500],monitor="testMonitor", units="deg")

#create some stimuli
grating = visual.GratingStim(win=mywin, size=1500, sf=2, ori=0)
fixation = visual.GratingStim(win=mywin, size=20, pos=[0,0], sf=0, rgb=-1)

#draw the stimuli and update the window
while True: #this creates a never-ending loop
    grating.setPhase(-0.2, '+')#advance phase by 0.05 of a cycle
    grating.draw()
    
    fixation.setPhase(-0.95, '+')#advance phase by 0.05 of a cycle
    fixation.draw()
    
    #fixation.draw()
    mywin.flip()

    if len(event.getKeys())>0: break
    event.clearEvents()

#cleanup
mywin.close()
core.quit()