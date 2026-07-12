# A CV-based Bandage for NHL Coordinate Data

## Problem

The NHL does not expose raw player coordinate data. You can get so much data from NHL EDGE, but
you can't get the (x, y) positions of players on the ice.

## Proposed solution

A computer-vision and machine-learning pipeline that watches a game broadcast
live and estimates player coordinates in rink-space.

It detects on-ice landmarks — lines and faceoff dots — and uses them to compute
a homography that maps broadcast-camera space onto a real rink coordinate
system. Player detections are then projected through that homography to recover
their positions on the ice.

## Current status
A testing harness was made to estimate whether homography would even be possible.
Players often times occlude on-ice landmarks, and a frame needs four landmarks to
be able to translate broadcast-space to rink-space. The test seemed reasonable, and
I am now building the actual homography computation engine.
