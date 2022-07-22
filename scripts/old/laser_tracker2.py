#!/usr/bin/env python
### FROM https://github.com/sanette/laser


class LaserTracker2(object):
    
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        
    def oneStepTracker(self, background, img, show, clipBox, snake, cal):
        mask = []

        if background.empty:
            background.add(img)
            return (mask, 0)
        
        diff, maxVal, maxLoc = diff_max(background.mean(), img, cal.laserDiameter//2)
        gm = globalMotion(diff, cal.motionThreshold)
        
        # We try to detect the pointer only if there is no global motion of the
        # image:
        if maxVal > cal.motionThreshold and gm < cal.globalMotionThreshold and insideBox(maxLoc, clipBox):
            # now we detect all the points above candidateThreshold and
            # try to select the best one...
            candidateThreshold = maxVal-5 # ?? ou motionThreshold ?
            candidates = maxValPos(diff, cal.motionThreshold, 10)
            printd (candidates.shape)
            if gdebug:
                for c in candidates:
                    printd (c)
                    plotVal(show, c)

            if not snake.empty():
                # we compute the predicted position
                p = snake.predict()
                printd ("Predicted = " + str(p))
                if gdebug:
                    cv2.circle(show, (p[0], p[1]), 10, PREDICTED_COLOR, 1)
            else:
                p = None

            # We select the candidate with the best score
            best, score = bestPixel(candidates, snake.active, snake.size, cal.jitterDist, cal.laserIntensity, p)
            printd ("SCORE = " + str(score))
            if snake.active:
                dd =  np.linalg.norm(p - best[0:2])
                printd ("Deviation from prediction = " + str(dd))
                # distance from last recorded point. Not used yet.
                d = np.linalg.norm(best[0:2] - snake.last())
                printd ("Distance = " + str(d))

            if gdebug:
                # TODO use the result of laserShape below! we could use the
                # fact that the shape often indicates the direction of the
                # pointer
                thr = cal.motionThreshold + (best[2] - cal.motionThreshold)/3
                # ou bien thr = motionThreshold ?
                mask, rect, angle = laserShape(diff, (best[0],best[1]), thr,
                                            maxRadius=int(cal.laserDiameter), debug=False)

            if score >= 0.5:
                printd ("==> Adding new point to snake.")
                newPoint = True # finally we register the best point

                if snake.size >= 1 and (np.linalg.norm(snake.last() - best[0:2]) > cal.laserDiameter):
                    background.add(img)
                else:
                    printd ("Not updating background because pointer did not move enough.")

                snake.grow(best[0:2])
                l = snake.length()
                a = snake.area(show)
                printd ("Length         = " + str(l))
                printd ("Area           = " + str(a))

            else:
                printd ("Nothing found.")

        snake.active = newPoint
        if (not snake.active) and snake.size > 0:
            snake.remove()

        printd ("Snake size = " + str(snake.size))
        if snake.size != 0:
            snake.draw(show)
        else:
            background.add(img)

        return (mask, maxVal)
        
    def diff_max(self, img1, img2, radius=2):
        """signed difference img2 - img1 and max value and position"""
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # We blur the images by radius at least 3. For 640x480 webcam, 3 seems to
        # substancially improve laser detection.
        radius = 2 * int((max(2, radius)) // 2) + 1 # should be an odd number
        gray1 = cv2.GaussianBlur(gray1, (radius, radius), 0)
        gray2 = cv2.GaussianBlur(gray2, (radius, radius), 0)
        h,w = gray1.shape
        diff = gray2.astype(int) - gray1.astype(int)
        imax = np.argmax(diff)
        x, y = (imax % w, imax // w)
        maxVal = diff[y,x]
        maxLoc = (x,y)
        return (diff, maxVal, maxLoc)
    
    # Here we detect "global motion", which is when we think the change in the
    # image is too important to be due to the laser pointer.
    #
    def globalMotion(self, gray, threshold):
        """percentage in [0,1] of moved pixels"""
        # This algorithm is efficient when there is little motion, which is the
        # case in principle most of the time.
        #
        # 0.000334 sec for globalMotion = 0.5
        # 0.000104 sec for globalMotion = 6e-06
        moved = gray[gray>threshold]
        h,w = gray.shape
        # pour obtenir leur position (linearisée): np.nonzero(gray > 10))
        return (len(moved) / float (w*h))

    def maxValPos(self, gray, threshold, nmax):
        """from the gray image, return an array of [x,y, value] with the higher values, of max length nmax"""

        selecty, selectx = np.where(gray >= threshold)
        # "np.where" is quite slow... about 10x more than "val=" or "sort" below...
        # typically 0.001 sec for 200 size
        val = gray[gray >= threshold] # let's hope the order is the same as that
                                    # was used for selectx/y...

        if len(selectx) <= nmax:
            res = np.column_stack((selectx,selecty,val))
            print_time ("stack", t0)
            return res

        else: # we need to sort... (this case should be avoided for performance)
            # we create a structured array in order to sort by the value :
            a = np.zeros((len(selectx),), dtype=[('x', 'i4'), ('y', 'i4'), ('val', 'i1')])
            a['x'] = selectx
            a['y'] = selecty
            a['val'] =  val
            sorted = np.sort(a, order='val')

            # we take the last nmax elements:
            best = sorted[-nmax:]

            # and convert back to a normal array:
            res = np.zeros((nmax,3), dtype=int)
            res[:,0] = best['x']
            res[:,1] = best['y']
            res[:,2] = best['val']
            print_time ("sorting", t0)
            return res
