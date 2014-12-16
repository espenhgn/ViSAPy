    '''Class for calculating the potential in a semi-infinite slice of neural tissue.
    Set-up:


              SALINE -> sigma_S = [sigma_Sx, sigma_Sy, sigma_Sz]

    <----------------------------------------------------> z = + a
    
              TISSUE -> sigma_T = [sigma_Tx, sigma_Ty, sigma_Tz]


                   o -> charge_pos = [x',y',z']


    <-----------*----------------------------------------> z = -a               
                 \-> elec_pos = [x,y,z] 

                 ELECTRODE -> sigma_G = [sigma_Gx, sigma_Gy, sigma_Gz]
        


    '''
