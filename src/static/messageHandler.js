// This script first creates listeners for the start and stop buttons.
// This script first creates listeners for the start and stop buttons.
// It also creates an EventSource that is linked to the url, '/generateData'. Once instantiated, the
// EventSource establishes a permanent socket that only allows server push communication.  In this
// application there are three event messages that are pushed to the browser: 'update', and
// 'jobfinished'.  There is also a listener for 'initialize', but it is not needed in this application.


let startPlotBtn = document.getElementById("startPlotBtn");
let stopPlotBtn = document.getElementById("stopPlotBtn");
startPlotBtn.addEventListener('click', startPlotProcess);
stopPlotBtn.addEventListener('click', stopPlotProcess);

let eventSourceGraph;

function startPlotProcess(){
    console.log("startProcess()");

    var regression_group_sizeOBJ   = document.getElementById("regression_group_size");
    var std_thresholdOBJ           = document.getElementById("std_threshold");
    var plot_scrolling_sizeOBJ     = document.getElementById("plot_scrolling_size");
    var pts_per_secOBJ             = document.getElementById("pts_per_sec");
    var selected_fileOBJ           = document.getElementById("dataFileNames");

    var regression_group_size   = regression_group_sizeOBJ.options[regression_group_sizeOBJ.selectedIndex].text;
    var std_threshold           = std_thresholdOBJ.options[std_thresholdOBJ.selectedIndex].text;
    var plot_scrolling_size     = plot_scrolling_sizeOBJ.options[plot_scrolling_sizeOBJ.selectedIndex].text;
    var pts_per_sec             = pts_per_secOBJ.options[pts_per_secOBJ.selectedIndex].text;
    var filename                = selected_fileOBJ.options[selected_fileOBJ.selectedIndex].text;

    initPlot();
    if(eventSourceGraph){
        eventSourceGraph.close();
    }
    // Create a JS EventSource object and give it the URL of a long running task.  The EventSource object
    // keeps the connection open to the given URL so that the process at the end point can send messages
    // back to the EventSource object.

    //const url = new URL('/generateData?');

     //pdm = PreprocessDataManager(regression_group_size,
     //                           plot_scrolling_size, col_name, anomaly_std_factor,
     //                           csv_file_name=file_name)

    const url = '/generateData?' + 'regression_size=' + regression_group_size + '&std_threshold=' + std_threshold +
        '&plot_scrolling_size=' + plot_scrolling_size + '&filename=' + filename + '&points_per_sec=' + pts_per_sec;
    console.log(url)
    const urlParams = new URLSearchParams(url);
    /*url.searchParams.append('regression_size', regression_group_size);
    url.searchParams.append('std_threshold', std_threshold);
    url.searchParams.append('plot_scrolling_size', plot_scrolling_size);
    url.searchParams.append('pts_per_sec', pts_per_sec);*/

    //finalURL = url.href;
    //console.log(url.href);

    eventSourceGraph = new EventSource(url);

    // NOTE:  This event 'initialize' is currently not used
    eventSourceGraph.addEventListener("initialize", function(event){
        initPlot();
    }, false);

    // "update" Event gets current job progress (how many iterations have been completed")
    eventSourceGraph.addEventListener("update", function(event){
        updatePlot(event.data);

    }, false);

    // "jobfinished" Event gets back finished message generated by the server when job finishes normally
    eventSourceGraph.addEventListener("jobfinished", function(event){

        console.log("Job finished, closing EventSource")
        eventSourceGraph.close();
        startPlotBtn.disabled = false;
    }, false);


    startPlotBtn.disabled = true;  // Disable start btn after plot is started.
}
// Closes the EventSource object which closes the connection between browser and servlet.  This
// puts the PrintWriter in the servelt to be in an error state.  On the server the error state is checked,
// and if true, closes the PrintWriter safely.
function stopPlotProcess(){
    console.log("Stopping process");
    eventSourceGraph.close();
    startPlotBtn.disabled = false;  // Enable start btn after process is terminated.
    //progressTextObj.innerHTML = "User Terminated Process";
}