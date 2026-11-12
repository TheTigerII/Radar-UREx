 # UREx Iteration Report Draft: Operator GUI and Radar Detection Range

> Working draft. Replace bracketed fields with your actual implementation, measurements, and dates. Do not report a range improvement until the paired test plan has been completed.

## Title

**Improving the Usability and Detection Range of a Portable COTS mmWave Radar for UAV Monitoring**

## Abstract (complete after the work)

This iteration extended the UREx portable mmWave radar prototype in two areas: an operator-facing graphical user interface (GUI) and radar detection range. The GUI was designed to [state intended operator tasks] and implemented using [framework / libraries] on [host device]. Range-related changes were made to [name processing stages, configuration parameters, or hardware changes], while [name unchanged settings] were held constant. We evaluated the prior implementation and the updated implementation using the same [radar profile], target (DJI Lito), target aspect, site, and range-test procedure. The primary endpoint was the longest tested range at which a confirmed radar track met the pre-registered trial success criterion. The prior implementation reached [X] m and the updated implementation reached [Y] m, a difference of [Y−X] m ([%]%). [Add uncertainty and false-alarm result.] These findings apply to the tested target and conditions and do not establish performance for other UAVs or environments.

## 1. Aim and research question

The earlier project implemented conventional range-Doppler and periodic micro-motion (PMM) processing on the IWR6843ISK-ODS and DCA1000EVM, with Jetson edge processing. It reported maximum static observations of 3.66 m and 3.95 m for the two approaches, respectively, but the report notes that the measurements lacked a documented common range-test protocol. The present iteration addresses a narrower engineering question with a controlled target and repeatable criterion.

**Research question.** Under a fixed radar profile and controlled target geometry, does the updated range-processing configuration increase the maximum repeatable detection range for a DJI Lito compared with the prior implementation, without an unacceptable increase in false tracks?

The GUI is a second deliverable. Its success is assessed through correct display of radar status and target state, clear presentation of range and confidence/track status, and operator ability to start, stop, and identify an invalid or unavailable stream. GUI usability is not treated as evidence of improved radar detection range.

## 2. Scope and contributions

### 2.1 My contributions (edit to reflect actual work)

I designed and implemented [GUI name/framework] for [Jetson / development computer]. Its main functions are [start/stop acquisition; select profile; show connection and capture integrity; visualize range-Doppler/point cloud/track; show range, velocity, and track state; save logs]. I connected the interface to the existing processing pipeline through [module/API/message format] and added [error handling, stale-data indication, settings validation, or other completed feature].

For range improvement, I investigated [observed range limitation], changed [specific algorithm or configuration], and kept [hardware, radar profile, antenna placement, and other controlled factors] fixed. The change was intended to improve range by [mechanism, e.g. increase coherent integration / adapt the detection threshold / reduce avoidable losses]. I authored and executed the matched baseline-versus-update test described in the separate test plan. My role in target operation, site setup, and data recording was [describe accurately]; assistance from [names/roles] was [describe].

### 2.2 Work completed by others / acknowledgements

[Name] assisted with [flight operation / safety / target positioning / radar setup / review]. The existing acquisition and processing code was developed in earlier project iterations; this work builds on it rather than claiming those earlier contributions. [Add supervisor and collaborator acknowledgements.] 

### 2.3 Deliverables

- GUI: [file/module, commit or version], providing [short feature list].
- Range update: [file/module, commit or version], changing [specific behavior].
- Evaluation: paired DJI Lito range-test log and summary at [path / date].
- Documentation: this report section and the separate test plan.

## 3. Design and implementation

### 3.1 GUI

The GUI presents [screens/panels] and receives data from [pipeline interface]. The displayed range is [measured / filtered / predicted] and is updated at [rate]. Track states are presented as [tentative / confirmed / coasting / lost, or actual states]. Capture integrity is represented by [packet/frame validity indicators]. The GUI indicates stale data after [duration] and reports [connection/configuration errors] without presenting old target data as current.

Design constraints included the Jetson's available compute, the existing bounded/latest-wins display transport, and keeping rendering work from blocking radar processing. [Describe design choice and any measured effect on frame timing, if measured.] A screenshot or diagram of the finished GUI is shown in Figure [X].

### 3.2 Range-related update

The limiting behavior observed in the baseline was [describe evidence: missed detections, weak return, thresholding, clutter, or other]. I changed [exact setting or code] from [baseline] to [updated] because [signal-processing rationale]. The expected effect was [specific prediction]. The following were held fixed during the comparison: radar hardware and placement, radar profile, antenna orientation, target, target operating mode, test site, processing output used for the endpoint, and trial duration. [List any factor that could not be held fixed.]

The GUI exposes [only validated settings / no signal-processing controls]. Its display and interaction changes are not counted as a signal-processing range improvement.

## 4. Evaluation method

We used the separate one-week test plan, version [X], on [date(s)] at [site]. The target was a DJI Lito [exact model/variant if known], operated [hovering / secured rotor-on fixture / other]. Target range was measured from [radar antenna reference point] to [target reference point] using [laser distance meter / tape / surveyed marks]. At each range, baseline and updated versions were tested in paired trials. The processing configuration was frozen before the confirmatory trials.

A trial counted as successful when [insert exact predeclared rule, e.g. a confirmed track was present in at least 80% of valid frames in a 30 s interval]. The maximum repeatable range was the farthest tested range with success in at least 4 of 5 trials. The primary comparison was the difference in maximum repeatable range. False-track behavior was checked during target-absent intervals using [same criterion].

## 5. Results (fill after testing)

### 5.1 GUI

The completed GUI supported [tasks]. In [N] observed operator sessions, [N] connection/configuration errors were correctly surfaced, [N] stale-stream events were identified, and [N] [other objective criterion] were completed. [If no structured usability evaluation was done, describe the features only and do not claim measured usability improvement.]

### 5.2 Range

| Measure | Baseline | Updated | Difference |
|---|---:|---:|---:|
| Maximum repeatable range (m) | [ ] | [ ] | [ ] |
| Successful trials at farthest qualifying range | [ ] / 5 | [ ] / 5 | — |
| Valid-frame detection coverage at that range | [ ]% | [ ]% | [ ] percentage points |
| Target-absent false-track trials | [ ] / [ ] | [ ] / [ ] | — |
| Capture/frame loss or invalid-frame rate | [ ] | [ ] | [ ] |

Figure [X] plots trial outcomes by range for each implementation. Individual trials are shown; frames within a trial are not treated as independent samples. [Add median/uncertainty summary if calculated.] The updated implementation [did/did not] meet the predeclared improvement criterion of [criterion].

## 6. Discussion

The observed range difference [supports / does not support / is inconclusive for] the hypothesis that [specific change] improves repeatable DJI Lito detection under the tested conditions. A change in maximum range is meaningful only alongside trial success and false-track results: [interpret measured outcome]. The result is specific to [target, aspect, mode, radar profile, environment, and software version]. The test does not establish performance against other drone models, at other aspects or altitudes, or in cluttered outdoor environments.

The GUI improved [documented workflow] by [evidence or specific capability]. It should be evaluated separately from radar detection performance because a clearer interface does not by itself change the radar's received signal or detection algorithm.

## 7. Limitations and next steps

This was a one-week introductory evaluation with [number] test sessions and one target model. Trials were [not fully independent / limited to one aspect / limited to one site]. Range was measured by [method and stated precision]. Wind, target motion, multipath, battery state, and small setup changes may affect results. The test provides an engineering comparison, not a broad operational-performance claim. Next steps are [repeat on another day/site; add other aspects/targets; quantify false alarms over longer target-absent periods; assess GUI usability with multiple operators].

## 8. Conclusion (complete after results)

This iteration delivered [GUI contribution] and evaluated [range-related change] against the previous implementation using a repeatable DJI Lito test. Under the stated setup, maximum repeatable range changed from [X] m to [Y] m, with [false-track/capture-integrity summary]. The result [does/does not/is not yet sufficient to] support the claim of improved range under these conditions. Broader UAV detection performance remains to be tested.

## Suggested report placement

Add this iteration as a new section or chapter after the prior system comparison, preserving the previous report as the baseline record. Update the abstract and conclusion only after the experiment is complete. Cite the separate test plan by title and version, and include the raw trial table or a link to it in an appendix.
