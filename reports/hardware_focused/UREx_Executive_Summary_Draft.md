# Portable COTS mmWave Radar for Small-Drone Detection

**Executive Summary of the UREx Project**

Ang Ding Li Victor  A0307828W  National University of Singapore  November 2026

## Project purpose

Usage of small UAVs have proliferated in recent years, creating a need for affordable solution to detect and monitor them. They are difficult radar targets because they are compact, can hover, and give only a weak reflection that varies as they turn. A radar with limited power and compute must also cope with clutter from the surroundings while staying portable and relatively affordable. This project sets out to show that such a radar could be built from off-the-shelf parts. It then attempts to find the maximum range such a setup can achieve.

## Approach and results

This project built and then improved a portable prototype radar that uses commercially available parts and an NVIDIA Jetson edge computer to detect and follow small drones. It ran in two stages. In the first stage, the team procured and assembled the components, developed two working analysis methods: a conventional method (Mode 1) and a rotor-sensitive method (Mode 2), then developed the classification model to distinguish between UAVs and non-UAV targets. In the second stage, the same hardware was pushed to detect targets further away, and a graphical interface was added to make the radar easier to set up and run.

The first stage produced a working, self-contained radar that runs from a swappable power bank on a camera tripod. It wasa able to reliably detect, track and classify a DJI Mini 4 quadcopter UAV at a maximum of 7 m away and correctly classify a human as a non UAV target out to the same range. In the second stage, the maximum detection and tracking range was extended to 11 m, showing a 4 m increase in range. Testing showed that target classification was unreliable past 7 m because the classifiers were trained only on data recorded with the first-stage radar settings. Hence, more data has to be collected to reliably classify targets at the new max range.

## Stage 1: building the portable radar

The prototype combines three main components:

- a Texas Instruments IWR6843ISK-ODS radar, operating at 60 to 64 GHz;
- a DCA1000EVM board that captures and processes raw data from the IWR board before sending it to the Jetson;
- an NVIDIA Jetson Orin Nano computer that runs the classification model and radar profiles.

These parts are mounted in a 3D-printed housing i designed in SolidWorks, which fits onto a standard camera tripod. A power bank slots into the housing and can be swapped quickly, so the radar can be carried and used away from mains power.

After i procured and integrated the components, Qi Xuan and Xian Yang, my groupmates, calibrated the radar using an E23012 corner reflector. Qi Xuan and i recorded radar data of a DJI Mini 4 and humans doing a range of motions to represent non UAV targets so Xian Yang could train the classification model. The conventional method developed by Qi Xuan finds targets by their distance and speed. The rotor-sensitive method instead looks for the small, regular movements of a drone's spinning rotors, which helps pick out a drone that is hovering. The team also calibrated the radar's distance readings, finding and correcting a fixed offset of about 12 cm for the conventional settings and 10 cm for the rotor-sensitive settings.

The team then measured how far away a DJI Mini 4 drone could be detected while it was held still or was hovering. The conventional method reached 6.5 m and the rotor-sensitive method reached 3.50 m. Covering the drone body with aluminium tape to strengthen its reflection raised these to 7 m and 3.95 m, which showed that the drone's weak reflection was one reason for the short range. Record logs show both methods processed every frame without data loss and gave the correct overall label in all test runs. The conventional method seems to be better since processed about three times as many frames per second with lower delay, and had almost double the range of the PMM method. However, these tests were a working-system check rather than a controlled comparison since the methods used different radar settings, were not run at the same time, and had no set speed and movement path of the drone and the human targets.

## Stage 2: extending detection range

The second stage attempted to increase the detection range of the radar while making it easier to operate. The work was done solely by me as my group mates had completed their parts and flew off for their exchange program.

A graphical operator interface and two extended-range modes, 3 and 4, were devloped and tested. The interface lets the operator choose a mode and its radar settings, start and stop a run, and check that the system is ready before recording. It also shows a top-down view of where the tracked target is relative to the radar and displays the path that the target travelled so the operator can follow a target without reading raw output.

Mode 3 uses a longer-range radar setting and combines all twelve receive channels to listen harder in one direction. It only confirms a target once it has appeared consistently over several frames. The trade-off is less precise distance readingsas it has a larger range bin than other modes. Mode 4 keeps Mode 1's radar profile and only changes how targets are picked out. It uses a more sensitive detection threshold, a cleaner estimate of the stationary background, all twelve receive channels combined, and a check for a target over several frames while preserving the same range bin.

Field testing was done with a DJI Lito X4, an updated version of the Mini 4 with almost identical physical dimensions. Results showed that both modes extended detection range beyond the 6.5 to 7 m reached in the first stage. Mode 4 performed best, detecting both the drone and a person out to 11 to 12 m, about 4-5 m further than the first stage. Mode 3 detected a person out to about 11 m, but the drone only out to about 9 m. Both modes ran in real time on the Jetson. In a 5.5-minute recording, for example, Mode 3 processed 3,279 frames at about 9.9 frames per second with no data loss. A typical frame took about 81 ms to process, within the radar's 100 ms frame interval, and fewer than 2% of frames missed that deadline.

Testing also exposed a false-target problem in Mode 3. When no one was in the scene, it reported a target about 2 or 6 m away and held onto it for most of the run and even after the targets left the detection zone. This was not caused by lost data. Strong reflections from stationary objects were leaking into the part of the signal used to find slow-moving targets.

Classification did not extend to the new ranges and was unreliable beyond about 7 m as classifiers were trained only on first-stage recordings, all made within about 7 m, so they have no examples of how targets look further away.

## Limitations

Currently, RustDesk is used to remotely access the jetson from a laptop as the jetson does not have a video output. This requires two devices to operate the system and both must be on the same wifi network to connect. This introduces considerable latency and increases the bulk of the system, reducing the portability and ease of use
The first-stage and second-stage results were measured with different radar settings and test setups, so they show the overall direction of progress rather than a direct like-for-like comparison. Efforts were made to better instrument the target characteristics for stage 2's tests but only 1 drone type was tested, and performance may differ with other UAVs.

## Recommended next steps

1. Procure a touchscreen portable monitor with a DP output so the system can be used standalone and does not require a second computer to operate through a remote viewing software.
2. Collect labelled drone and non-drone recordings with the extended-range modes, including at the new distances with a larger variety of drone target sizes, and retrain the classifiers. Keep entire recording sessions and locations in separate training and test groups.
3. Repeat matched trials at measured distances for both the original and extended-range modes, using the same drone. Record an independent reference position so that range, tracking and timing can be compared on the same basis.

## Conclusion

This project showed that a portable radar built from off-the-shelf parts can detect and track small drones in real time on an edge computer. With rudimentary hardware like the IWR board a range of 11-12m was achieved, so it is not a stretch of an imagination to say that arrays with more power or TX/RX elements at a similar form factor can achieve much higher range with the correct signal processing methods. An instrumented range with a clean background would have significantly aided the quality of the collected results along with a more rigourous test plan spanning more target types to increase the capability of the classifier. 




