function getGuid() {
    function createGuid() {
        return (((1 + Math.random()) * 65536) | 0).toString(16).substring(1)
    }
    guid1 = createGuid() + createGuid() + "-" + createGuid() + "-" + createGuid() + createGuid() + "-" + createGuid() + createGuid() + createGuid();
    return guid1
}