# persistent_location_extraction
My internship work in summer 2015 for Newfield Wireless, a Berkeley based company: Project to extract mobility insights and find persistent location for over a million cell-phone users of a tier-1 US telecom operator. "Persistent location" is one where the user spends most of his time (may or may not be his home).

The core hypothesis of this project was that the time period of reduced user mobility corresponded to the user being at his persistent location. Of course there are challenges to this (as outlined in the ppt uploaded on my linkedIn) that need to be considered.

The script "dls_reader" is where I played with user's mobility patterns to see what if any statistical properties can be observed to hold, say with regards to distance travelled before generating a new location on the move, mean time between generation of new locations etc. The original LTE data was stored in csv files, having millions of entries with many fields for every entry (user-id, device-id, start time, end time, latitude, longitude, error in measurement, number of signal measurements taken, whether msg/call, time-zone etc.)

To model their mobility patterns, I applied curve-fitting, local minima analysis and weighted k-means clustering algorithms. Each of these approaches had its own advantages and shortcomings and the same were studied.

Curve-fitting was used because the mobility plot of the median user suggested so and it seemed intuitive. Plotting the mean number of locations generated over a period of 3 days also showed characteristics of being modelled by certain curve. So, it was implemented using scipy library and I tried simple curves like binomial, gaussian, beta, gamma etc. but due to sparsity in data sometimes, their goodness of fit obatined using Pearson chi-square test wasn't good. I haven't included the script for that because not much was done by me there, except write the gooddness of fit part and try different curves. I concluded that part with an understanding that a mixture of curves would be a better way to go although it would take more compute power to model.

The explanation of other approaches is given in their folders respectively. The conclusion at the end of the project was that local minima analysis performed better than other two for our purpose. For the median user, it predicted the persistent location to within 200 m. accuracy using call records of only the recent most 2 days (good accuracy for less data and more simple in implementation). 

But the weighted k-means approach not only gave the persistent location info, but also told us the typical time of day a user would be at his persistent location and how long he tends to stay there. But it needs more data for good performance and because of the way "weighting" was implemented, some post-processing of the results needs to be done.

P.S. I worked on this project solo and I was guided by Michael Ellenberg, my supervisor during the internship. The code was written by me and the visible improvement in writing the same happened over time, thanks to Michael's insightful tips. If someone does not follow why certain algorithm was used or why was it implemented the way it was, please feel free to ask me (my profile bears my email-id).
