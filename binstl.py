import struct
import os
import argparse

def main(infilename, outfilename):
    filename = os.path.basename(infilename)
    print(filename)
    with open(infilename, "rb") as f:
        data = f.read()
    
    # STL Header        
    header = data[:80]

    # Length of Surfaces
    bin_faces = data[80:84]
    faces = struct.unpack('I',bin_faces)[0]

    # Create ASCII STL
    out = open(outfilename, 'w') 
    out.write(f"solid {filename}\n")
    for x in range(0, faces):
        out.write("facet normal ")

        xc = data[84+x*50 : (84+x*50)+4]
        yc = data[88+x*50 : (88+x*50)+4]
        zc = data[92+x*50 : (92+x*50)+4]

        out.write(str(struct.unpack('f',xc)[0]) + " ")
        out.write(str(struct.unpack('f',yc)[0]) + " ")
        out.write(str(struct.unpack('f',zc)[0]) + "\n")

        out.write(" outer loop\n")
        for y in range(1,4):
            out.write("  vertex ")

            xc = data[84+y*12+x*50 : (84+y*12+x*50)+4]
            yc = data[88+y*12+x*50 : (88+y*12+x*50)+4]
            zc = data[92+y*12+x*50 : (92+y*12+x*50)+4]
            
            out.write(str(struct.unpack('f',xc)[0]) + " ")
            out.write(str(struct.unpack('f',yc)[0]) + " ")
            out.write(str(struct.unpack('f',zc)[0]) + "\n")

        out.write(" endloop\n")
        out.write("endfacet\n")
        
    out.write(f"endsolid {filename}\n")
    out.close()

if __name__ == '__main__':
    program_args = argparse.ArgumentParser(description='Bin to ASCII STL Converter')
    program_args.add_argument('-i', '--inputfile' , required=True,  help="Input file path")
    program_args.add_argument('-o', '--outputfile', required=False, help="Output file path")
    args = program_args.parse_args()

    if args.inputfile:
        infilename = args.inputfile
        
    if args.outputfile:
        outfilename = args.outputfile
        assert ".stl" in outfilename.lower(), 'unsupported file : "it supports .STL File Format"'
    else:
        outfilename = os.path.join(os.path.dirname(infilename), f"ASCII-{os.path.basename(infilename)}")
    
    os.makedirs(os.path.dirname(outfilename), exist_ok=True)

    main(infilename, outfilename)