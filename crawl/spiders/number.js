function getNumber(url) {
    var nyzm = url.indexOf("&number");
    var subyzm = url.substring(nyzm + 1);
    var yzm1 = subyzm.substr(7, 4);
    return yzm1
}