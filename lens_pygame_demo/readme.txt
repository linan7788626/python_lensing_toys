date: 24.04.2006 

project: sprite engine

who: DR0ID
	mail: dr0id@bluewin.ch
	homepage: http://www.mypage.bluewin.ch/DR0ID/index.html 
	
artwork: DR0ID

how to run: test_lens.py
			
			space: toggle lens attachement to mouse
			p: toggle attach lens to moving sprite(invisible)
			c: clear's lens from screen
			d: toggle the debug modus of sprite engine (1 fps so one can see what happens)
				colors of rects mean:
								red: screen updated area
								green: new position of dirty sprite
								orange: old position of dirty rect
								blue: colliding with dirty screen area, so re-blitted
								yellow: lost rect (removed from renderer)
								
licenses: ?

dependecies: pyhton (www.python.org), pygame (www.pygame.org)

comments:	It is still under heavy developement
			
			In many cases (games, apps) only a few sprites are dirty and need a redraw...
			pygame.sprite module clears and redraws all sprites in the group.
			It uses dirty sprites to unly redraw the dirty parts. If all sprites are dirty then the 
			performance should not be slower than pygame.sprite (~equal more or less)