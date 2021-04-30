(function (w) {
w.URLSearchParams = w.URLSearchParams || function (searchString) {
  var self = this;
  self.searchString = searchString;
  self.get = function (name) {
    var results = new RegExp('[\?&]' + name + '=([^&#]*)').exec(self.searchString);
    if (results === null) {
      return null;
    }
    else {
      return decodeURI(results[1]) || 0;
    }
  };
}
})(window);

const queryString = window.location.search;
const urlParams = new URLSearchParams(queryString);
const lang = urlParams.get('lang')
window.onload = function () {
var domainName = window.location.hostname;

// remove an instance of shortbread if already exists
var existingShortbreadEl = document.getElementById("awsccc-sb-ux-c");
existingShortbreadEl && existingShortbreadEl.remove();        

var shortbread = AWSCShortbread({
  domain: domainName,
  language: lang,
  //queryGeolocation: function (geolocatedIn) { geolocatedIn("EU") },
});

shortbread.checkForCookieConsent();
}
