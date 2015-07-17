CC       	=	cc
CFLAGS	 	=	-Wall -g -O3
GSL_INCL 	=	-I/usr/local/include/
GSL_LIBS 	=	-L/usr/local/lib/
FFTW_INCL	=	-I/home/uranus/software/include/ 
FFTW_LIBS	=	-L/home/uranus/software/lib/
         	
         	
#EXEC  	 	=	theta_alpha
#OBJS  	 	=	theta_alpha.o my_func.o
#SOURCES 	=	theta_alpha.c my_func.h my_func.c
         	
EXEC  	 	=	youkme
OBJS  	 	=	youkme.o my_func.o
SOURCES 	=	youkme.c my_func.h my_func.c
         	
#EXEC  	 	=	surface_fftw
#OBJS  	 	=	surface_fftw.o my_func.o
#SOURCES 	=	surface_fftw.c my_func.h my_func.c

#EXEC  	 	=	psi
#OBJS  	 	=	psi.o my_func.o
#SOURCES 	=	psi.c my_func.h my_func.c
         	
GSL_LIB  	=	-lgsl -lgslcblas
FFTW_LIB 	=	-lfftw3 -lm
         	
INCL		=	$(GSL_INCL) $(FFTW_INCL)         	
LIBS   	 	=	$(GSL_LIBS) $(GSL_LIB) \
			$(FFTW_LIBS) $(FFTW_LIB) 
#-------------------------------------------
	
$(EXEC): $(OBJS) 
	$(CC) $(OBJS) $(CFLAGS) $(INCL) $(LIBS) -o  $(EXEC)  

$(OBJS): %.o:%.c
	$(CC) -c $(CFLAGS) $(INCL) $< -o  $@  

clean:
	rm -f $(OBJS) $(EXEC)
