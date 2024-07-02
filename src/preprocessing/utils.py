
def get_time_window(peri_stim_time_win=None):
    """
    Get the tmin,tmax,bmax for any custom time window.
    Also get the custom save path.
    """
    bmax = 0.0
    if peri_stim_time_win is None:
        t_win = float(
            input(
                "Please enter the peri-stimulus time window."
                + "\nEx: '0 (default)' = [-0.2,0.8], '2' = [-1.0,1.0], etc...\n\n>> "
            )
        )
    else:
        t_win = float(peri_stim_time_win)

    if t_win == 0.0:
        tmin, tmax = -0.2, 0.8
        time_win_path = ""
    else:
        tmin, tmax = -t_win / 2, t_win / 2
    print(f"[{tmin},{bmax},{tmax}]")
    time_win_path = f"{int(t_win)}_sec_time_window/"
    # print(time_win_path)
    return (tmin, bmax, tmax), time_win_path

