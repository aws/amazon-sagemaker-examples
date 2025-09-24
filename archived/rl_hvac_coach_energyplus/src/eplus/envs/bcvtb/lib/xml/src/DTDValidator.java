import java.io.File;
import java.io.IOException;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.ParserConfigurationException;
import org.w3c.dom.Document;
import org.xml.sax.SAXException;
import org.xml.sax.SAXParseException;
import org.xml.sax.ErrorHandler;

/** Class to validate xml files.
 *
 * @author Michael Wetter
 * @since BCVTB 0.3
 */

public class DTDValidator {
    /** The xml file name */
    static String fil;

    /** Main method.
     *
     *@param args Argument list, args[0] must be the xml file name
     *            that will be parsed, arg[1] must be the path to the dtd file.
     */
    public static void main(String[] args) {
	fil = args[0];
	// Document builder factory
	DocumentBuilderFactory fac = DocumentBuilderFactory.newInstance();
	fac.setValidating(true);
	try {
	    // Document builder and error handler
	    DocumentBuilder b = fac.newDocumentBuilder();
	    b.setErrorHandler(new MyErrorHandler());
	    // Parse document
	    //	    Document d = b.parse(args[0]);
	    Document d = b.parse(new java.io.FileInputStream(fil), 
				 args[1] + File.separator);
	} 
	catch(SAXException e) {
	    System.err.println(e.getMessage());
	}
	catch(IOException e) {
	    System.err.println(e.getMessage());
	}
	catch (ParserConfigurationException e) {
	    System.err.println(e.getMessage());
	}
    }
    /** Inner class for error handling
     */
    private static class MyErrorHandler implements ErrorHandler {

	/** Writes warning messages to <code>System.err</code>
	 *
	 *@param e The exception 
	 *@exception SAXException If a SAXException occurs
	 */
	public void warning(SAXParseException e) throws SAXException {
	    printInfo("Warning: ", e);
	    System.exit(1);
	}

	/** Writes error messages to <code>System.err</code>
	 *
	 *@param e The exception 
	 *@exception SAXException If a SAXException occurs
	 */
	public void error(SAXParseException e) throws SAXException {
	    printInfo("Error: ", e);
	    System.exit(1);
	}

	/** Writes error messages to <code>System.err</code>
	 *
	 *@param e The exception 
	 *@exception SAXException If a SAXException occurs
	 */
	public void fatalError(SAXParseException e) throws SAXException {
	    error(e);
	}


	/** Prints the error message to <code>System.err</code>
	 *@param s The string that will be added in front of the exception
	 *         message
	 *@param e The exception
	 */
	private void printInfo(String s, SAXParseException e) {
	    System.err.println(s + 
			       fil + ":"
			       + e.getLineNumber() + ": " 
			       + e.getMessage());
	}
    }
}
/********************************************************************
Copyright Notice
----------------

Building Controls Virtual Test Bed (BCVTB) Copyright (c) 2008-2009, The
Regents of the University of California, through Lawrence Berkeley
National Laboratory (subject to receipt of any required approvals from
the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this
software, please contact Berkeley Lab's Technology Transfer Department
at TTD@lbl.gov

NOTICE.  This software was developed under partial funding from the U.S.
Department of Energy.  As such, the U.S. Government has been granted for
itself and others acting on its behalf a paid-up, nonexclusive,
irrevocable, worldwide license in the Software to reproduce, prepare
derivative works, and perform publicly and display publicly.  Beginning
five (5) years after the date permission to assert copyright is obtained
from the U.S. Department of Energy, and subject to any subsequent five
(5) year renewals, the U.S. Government is granted for itself and others
acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide
license in the Software to reproduce, prepare derivative works,
distribute copies to the public, perform publicly and display publicly,
and to permit others to do so.


Modified BSD License agreement
------------------------------

Building Controls Virtual Test Bed (BCVTB) Copyright (c) 2008-2009, The
Regents of the University of California, through Lawrence Berkeley
National Laboratory (subject to receipt of any required approvals from
the U.S. Dept. of Energy).  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
   2. Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.
   3. Neither the name of the University of California, Lawrence
      Berkeley National Laboratory, U.S. Dept. of Energy nor the names
      of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

You are under no obligation whatsoever to provide any bug fixes,
patches, or upgrades to the features, functionality or performance of
the source code ("Enhancements") to anyone; however, if you choose to
make your Enhancements available either publicly, or directly to
Lawrence Berkeley National Laboratory, without imposing a separate
written license agreement for such Enhancements, then you hereby grant
the following license: a non-exclusive, royalty-free perpetual license
to install, use, modify, prepare derivative works, incorporate into
other computer software, distribute, and sublicense such enhancements or
derivative works thereof, in binary and source code form.

********************************************************************
*/
