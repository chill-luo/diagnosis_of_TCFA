import os
import time
import torch
import xml.etree.ElementTree as ET


def mask_vis(mask):
    sample = mask
    temp = torch.zeros_like(sample)
    temp[sample == 3] = 255
    res = temp.unsqueeze(0)
    temp = torch.zeros_like(sample)
    temp[sample == 2] = 255
    temp[sample == 3] = 255
    temp = temp.unsqueeze(0)
    res = torch.cat((res, temp), dim=0)
    temp = torch.zeros_like(sample)
    temp[sample == 1] = 255
    temp[sample == 3] = 255
    temp = temp.unsqueeze(0)
    res = torch.cat((res, temp), dim=0)

    return res


class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)
    

class XMLParser():
    def __init__(self, file=None, root_name='root'):
        if file is not None:
            assert os.path.exists(file)
            self.tree = ET.ElementTree()
            self.tree.parse(file)
        else:
            root = ET.Element(root_name)
            self.tree = ET.ElementTree(root)
        self.root = self.tree.getroot()
        self.root_name = self.root.tag
        
    def __len__(self):
        return len(self.tree)

    def __root__(self):
        return self.tree.getroot()

    def _find_with_id(self, node, child):
        name = child.split(':')[0]
        id = child.split(':')[1] if len(child.split(':')) == 2 else None
        children = node.findall(name)
        for c in children:
            if c.get('id') != id:
                continue
            else:
                return c
        print(f'Something wrong with node {child}')
        return

    def _pretty(self, element, indent='\t', newline='\n', level = 0): 
        if element: 
            if element.text == None or element.text.isspace():     
                element.text = newline + indent * (level + 1)      
            else:    
                element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
        temp = list(element) 
        for subelement in temp:    
            if temp.index(subelement) < (len(temp) - 1):     
                subelement.tail = newline + indent * (level + 1)      
                subelement.tail = newline + indent * level
            self._pretty(subelement, indent, newline, level = level + 1)
        return

    def save(self, path='./tmp.xml'):
        self._pretty(self.root)
        self.tree.write(path)
        return

    def add_node(self, node_path, content=None):
        '''
        if you want to visit path with special id,
        you could input path as follow:
            path = 'child/grandchild:jack' 
        or:
            path = 'child:tom/grandchild' 
        '''
        element = self.root
        elements = node_path.split('/')
        target = elements[-1]
        if len(target.split(':')) == 1:
            new = ET.Element(target)
        else:
            node = target.split(':')[0]
            id = target.split(':')[1]
            new = ET.Element(node, {'id': id})
        if content is not None:
            new.text = str(content)
        if len(elements) > 1:
            elements = elements[:-1]
            for e in elements:
                element = self._find_with_id(element, e)
        element.append(new)
        return

    def remove_node(self, node_path):
        element = self.root
        elements = node_path.split('/')
        target = elements[-1]
        if len(elements) > 1:
            elements = elements[:-1]
            for e in elements:
                element = self._find_with_id(element, e)
        target = self._find_with_id(element, target)
        element.remove(target)
        return

    def read_node(self, node_path ,just_tree=False):
        element = self.root
        elements = node_path.split('/')
        if len(elements) > 1:
            elements = elements[:-1]
            for e in elements:
                element = self._find_with_id(element, e)
        if (element.text == None) or just_tree:
            return ET.tostring(element, encoding='unicode')
        else:
            return element.text